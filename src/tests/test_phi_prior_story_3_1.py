"""
Story 3.1: Asset-Class Adaptive phi Prior -- Tests
===================================================
Tests for:
  1. _classify_asset_for_phi(): correct classification of symbols
  2. compute_phi_prior(): correct (phi_0, lambda_phi) per asset class
  3. Integration into Student-t Stage 1 (phi_prior_center/lambda params)
  4. Integration into Gaussian Stage 1 (phi_prior_center/lambda params)
  5. Cross-asset phi pooling stratification in tune.py
"""
import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestClassifyAssetForPhi(unittest.TestCase):
    """Test _classify_asset_for_phi returns correct asset class."""

    def setUp(self):
        from models.phi_student_t_unified import _classify_asset_for_phi
        self.classify = _classify_asset_for_phi

    def test_index_symbols(self):
        for sym in ['SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI']:
            self.assertEqual(self.classify(sym), 'index', f"{sym} should be index")

    def test_large_cap_symbols(self):
        for sym in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM']:
            self.assertEqual(self.classify(sym), 'large_cap', f"{sym} should be large_cap")

    def test_small_cap_symbols(self):
        for sym in ['UPST', 'AFRM', 'IONQ', 'SNAP', 'DKNG']:
            self.assertEqual(self.classify(sym), 'small_cap', f"{sym} should be small_cap")

    def test_crypto_symbols(self):
        for sym in ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD']:
            self.assertEqual(self.classify(sym), 'crypto', f"{sym} should be crypto")

    def test_metals_symbols(self):
        for sym in ['GC=F', 'SI=F', 'GLD', 'SLV', 'HG=F', 'PL=F']:
            self.assertEqual(self.classify(sym), 'metals', f"{sym} should be metals")

    def test_forex_symbols(self):
        for sym in ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']:
            self.assertEqual(self.classify(sym), 'forex', f"{sym} should be forex")

    def test_high_vol_symbols(self):
        for sym in ['MSTR', 'SMCI', 'RGTI', 'QBTS']:
            self.assertEqual(self.classify(sym), 'high_vol', f"{sym} should be high_vol")

    def test_unknown_default(self):
        self.assertEqual(self.classify('UNKNOWN_TICKER_XYZ'), 'default')
        self.assertEqual(self.classify('ACME'), 'default')

    def test_none_returns_default(self):
        self.assertEqual(self.classify(None), 'default')

    def test_case_insensitive(self):
        # classify uppercases internally
        self.assertEqual(self.classify('spy'), 'index')
        self.assertEqual(self.classify('aapl'), 'large_cap')

    def test_futures_fallback_to_metals(self):
        # Any =F symbol not in named sets falls through to metals via endswith('=F')
        self.assertEqual(self.classify('ZZ=F'), 'metals')


class TestComputePhiPrior(unittest.TestCase):
    """Test compute_phi_prior returns correct (phi_0, lambda_phi)."""

    def setUp(self):
        from models.phi_student_t_unified import (
            compute_phi_prior,
            PHI_PRIOR_INDEX,
            PHI_PRIOR_LARGE_CAP,
            PHI_PRIOR_SMALL_CAP,
            PHI_PRIOR_CRYPTO,
            PHI_PRIOR_METALS,
            PHI_PRIOR_FOREX,
            PHI_PRIOR_HIGH_VOL,
            PHI_PRIOR_DEFAULT,
            PHI_PRIOR_LAMBDA_BASE,
        )
        self.compute = compute_phi_prior
        self.expected = {
            'SPY': PHI_PRIOR_INDEX,         # 0.95
            'AAPL': PHI_PRIOR_LARGE_CAP,    # 0.80
            'UPST': PHI_PRIOR_SMALL_CAP,    # 0.30
            'BTC-USD': PHI_PRIOR_CRYPTO,    # 0.70
            'GC=F': PHI_PRIOR_METALS,       # 0.85
            'EURUSD=X': PHI_PRIOR_FOREX,    # 0.50
            'MSTR': PHI_PRIOR_HIGH_VOL,     # 0.20
            'UNKNOWN': PHI_PRIOR_DEFAULT,   # 0.50
        }
        self.lambda_base = PHI_PRIOR_LAMBDA_BASE

    def test_class_specific_phi_0_no_returns(self):
        """Without returns, phi_0 should match class prior exactly."""
        for sym, expected_phi in self.expected.items():
            phi_0, _ = self.compute(sym, returns=None, n_samples=252)
            self.assertAlmostEqual(phi_0, expected_phi, places=4,
                                   msg=f"{sym}: phi_0={phi_0}, expected={expected_phi}")

    def test_spy_phi_near_0_95(self):
        phi_0, _ = self.compute('SPY', returns=None)
        self.assertGreater(phi_0, 0.90)
        self.assertLessEqual(phi_0, 0.99)

    def test_upst_phi_near_0_30(self):
        phi_0, _ = self.compute('UPST', returns=None)
        self.assertLess(phi_0, 0.50)

    def test_lambda_base_at_252(self):
        """At n_samples=252, lambda = lambda_base."""
        _, lam = self.compute('SPY', returns=None, n_samples=252)
        self.assertAlmostEqual(lam, self.lambda_base, places=6)

    def test_lambda_inversely_proportional_to_n(self):
        """Larger sample -> smaller lambda (weaker shrinkage)."""
        _, lam_short = self.compute('SPY', returns=None, n_samples=100)
        _, lam_long = self.compute('SPY', returns=None, n_samples=500)
        self.assertGreater(lam_short, lam_long)

    def test_lambda_floor_at_n50(self):
        """n_samples < 50 is floored at 50."""
        _, lam_10 = self.compute('SPY', returns=None, n_samples=10)
        _, lam_50 = self.compute('SPY', returns=None, n_samples=50)
        self.assertAlmostEqual(lam_10, lam_50, places=6)

    def test_acf_blending_with_returns(self):
        """With returns that have known ACF, phi_0 shifts toward ACF(1)."""
        np.random.seed(42)
        # Generate AR(1) series with phi_true=0.8
        n = 500
        r = np.zeros(n)
        for i in range(1, n):
            r[i] = 0.8 * r[i - 1] + np.random.normal(0, 0.01)

        # For SPY (class prior=0.95), ACF should be ~0.8
        # Blended: 0.7*0.95 + 0.3*acf1 -> should be < 0.95
        phi_0_no_ret, _ = self.compute('SPY', returns=None, n_samples=n)
        phi_0_with_ret, _ = self.compute('SPY', returns=r, n_samples=n)
        # ACF < 0.95 => blended phi should be < class prior
        self.assertLess(phi_0_with_ret, phi_0_no_ret)

    def test_acf_blending_high_momentum(self):
        """If ACF > class prior, phi_0 should increase."""
        np.random.seed(123)
        n = 500
        r = np.zeros(n)
        for i in range(1, n):
            r[i] = 0.95 * r[i - 1] + np.random.normal(0, 0.01)

        # For UPST (class prior=0.30), ACF~0.95 should pull phi_0 up
        phi_0_no_ret, _ = self.compute('UPST', returns=None, n_samples=n)
        phi_0_with_ret, _ = self.compute('UPST', returns=r, n_samples=n)
        self.assertGreater(phi_0_with_ret, phi_0_no_ret)

    def test_phi_0_bounded(self):
        """phi_0 always in [-0.99, 0.99]."""
        for sym in ['SPY', 'UPST', 'BTC-USD', 'GC=F', 'MSTR', 'UNKNOWN']:
            phi_0, _ = self.compute(sym, returns=None)
            self.assertGreaterEqual(phi_0, -0.99)
            self.assertLessEqual(phi_0, 0.99)

    def test_returns_with_nans_handled(self):
        """NaN values in returns should not crash."""
        r = np.array([0.01, -0.02, np.nan, 0.005, -0.01, np.nan, 0.02] * 20)
        phi_0, lam = self.compute('SPY', returns=r, n_samples=len(r))
        self.assertTrue(np.isfinite(phi_0))
        self.assertTrue(np.isfinite(lam))

    def test_short_returns_fallback(self):
        """< 50 valid returns -> no ACF blending, just class prior."""
        r = np.random.normal(0, 0.01, 30)
        phi_0_short, _ = self.compute('SPY', returns=r, n_samples=30)
        phi_0_none, _ = self.compute('SPY', returns=None, n_samples=30)
        # Should be the same since ACF not computed for < 50 valid
        self.assertAlmostEqual(phi_0_short, phi_0_none, places=4)


class TestStudentTIntegration(unittest.TestCase):
    """Test phi prior is threaded into Student-t _stage_1_base_params."""

    def test_stage1_accepts_phi_prior_params(self):
        """_stage_1_base_params should accept phi_prior_center/lambda kwargs."""
        from models.phi_student_t_unified import UnifiedPhiStudentTModel
        import inspect
        sig = inspect.signature(UnifiedPhiStudentTModel._stage_1_base_params)
        param_names = list(sig.parameters.keys())
        self.assertIn('phi_prior_center', param_names)
        self.assertIn('phi_prior_lambda', param_names)

    def test_optimize_with_different_assets(self):
        """optimize_params_unified produces different phi for SPY vs UPST."""
        from models.phi_student_t_unified import UnifiedPhiStudentTModel
        np.random.seed(42)
        n = 300
        returns = np.random.normal(0.0003, 0.015, n)
        vol = np.abs(returns) * 0.5 + 0.01

        config_spy, diag_spy = UnifiedPhiStudentTModel.optimize_params_unified(
            returns=returns, vol=vol, nu_base=8.0, asset_symbol='SPY',
        )
        config_upst, diag_upst = UnifiedPhiStudentTModel.optimize_params_unified(
            returns=returns, vol=vol, nu_base=8.0, asset_symbol='UPST',
        )
        # Both should succeed
        self.assertIsNotNone(config_spy)
        self.assertIsNotNone(config_upst)

        # Extract phi from config
        phi_spy = config_spy.phi
        phi_upst = config_upst.phi
        self.assertIsNotNone(phi_spy, "SPY result missing phi")
        self.assertIsNotNone(phi_upst, "UPST result missing phi")

        # With same data but different priors, phi should differ
        # SPY prior=0.95 pulls up; UPST prior=0.30 pulls down
        self.assertNotAlmostEqual(phi_spy, phi_upst, places=2,
                                  msg=f"SPY phi={phi_spy}, UPST phi={phi_upst} should differ")

    def test_optimize_spy_phi_higher_than_upst(self):
        """SPY's phi prior is higher, so its phi estimate should be higher."""
        from models.phi_student_t_unified import UnifiedPhiStudentTModel
        np.random.seed(99)
        n = 300
        returns = np.random.normal(0.0002, 0.012, n)
        vol = np.abs(returns) * 0.4 + 0.01

        config_spy, _ = UnifiedPhiStudentTModel.optimize_params_unified(
            returns=returns, vol=vol, nu_base=8.0, asset_symbol='SPY',
        )
        config_upst, _ = UnifiedPhiStudentTModel.optimize_params_unified(
            returns=returns, vol=vol, nu_base=8.0, asset_symbol='UPST',
        )
        phi_spy = config_spy.phi
        phi_upst = config_upst.phi

        self.assertGreater(phi_spy, phi_upst,
                           f"SPY phi={phi_spy} should > UPST phi={phi_upst}")


class TestGaussianIntegration(unittest.TestCase):
    """Test phi prior is threaded into Gaussian _gaussian_stage_1."""

    def test_stage1_accepts_phi_prior_params(self):
        """_gaussian_stage_1 should accept phi_prior_center/lambda kwargs."""
        from models.gaussian import GaussianDriftModel
        import inspect
        sig = inspect.signature(GaussianDriftModel._gaussian_stage_1)
        param_names = list(sig.parameters.keys())
        self.assertIn('phi_prior_center', param_names)
        self.assertIn('phi_prior_lambda', param_names)

    def test_optimize_with_different_assets(self):
        """Gaussian optimize produces different phi for SPY vs MSTR."""
        from models.gaussian import GaussianDriftModel
        np.random.seed(42)
        n = 300
        returns = np.random.normal(0.0003, 0.015, n)
        vol = np.abs(returns) * 0.5 + 0.01

        config_spy, diag_spy = GaussianDriftModel.optimize_params_unified(
            returns=returns, vol=vol, asset_symbol='SPY',
        )
        config_mstr, diag_mstr = GaussianDriftModel.optimize_params_unified(
            returns=returns, vol=vol, asset_symbol='MSTR',
        )
        self.assertIsNotNone(config_spy)
        self.assertIsNotNone(config_mstr)

        phi_spy = config_spy.phi
        phi_mstr = config_mstr.phi
        if phi_spy is not None and phi_mstr is not None:
            # SPY prior=0.95, MSTR prior=0.20 => should differ
            self.assertNotAlmostEqual(phi_spy, phi_mstr, places=2,
                                      msg=f"SPY phi={phi_spy}, MSTR phi={phi_mstr}")


class TestCrossAssetPhiPooling(unittest.TestCase):
    """Test apply_cross_asset_phi_pooling is stratified by asset class."""

    def _build_cache(self, entries):
        """Build a minimal cache dict from list of (asset, phi, n_samples)."""
        cache = {}
        for asset, phi, n_samples in entries:
            cache[asset] = {
                "global": {"phi": phi},
                "data_length": n_samples,
            }
        return cache

    def test_stratified_pooling_preserves_class_separation(self):
        """Index assets should pool toward index median, not toward small-cap median."""
        from tuning.tune import apply_cross_asset_phi_pooling

        entries = []
        # 6 index assets with phi ~ 0.90
        for i, sym in enumerate(['SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI']):
            entries.append((sym, 0.88 + 0.02 * i, 500))
        # 6 small-cap assets with phi ~ 0.25
        for i, sym in enumerate(['UPST', 'AFRM', 'IONQ', 'SNAP', 'DKNG', 'CRWD']):
            entries.append((sym, 0.20 + 0.02 * i, 500))

        cache = self._build_cache(entries)
        result = apply_cross_asset_phi_pooling(cache)

        # Index phi values should still be close to 0.90 (not dragged to ~0.55 global median)
        spy_phi = result['SPY']['global']['phi']
        self.assertGreater(spy_phi, 0.80,
                           f"SPY phi={spy_phi} should remain high (pooled within index class)")

        # Small-cap phi should stay low
        upst_phi = result['UPST']['global']['phi']
        self.assertLess(upst_phi, 0.50,
                        f"UPST phi={upst_phi} should remain low (pooled within small-cap class)")

    def test_metadata_includes_class_stats(self):
        """Pooling metadata should include per-class statistics."""
        from tuning.tune import apply_cross_asset_phi_pooling

        entries = []
        for i, sym in enumerate(['SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI']):
            entries.append((sym, 0.90, 500))
        for i, sym in enumerate(['UPST', 'AFRM', 'IONQ', 'SNAP', 'DKNG', 'CRWD']):
            entries.append((sym, 0.25, 500))

        cache = self._build_cache(entries)
        result = apply_cross_asset_phi_pooling(cache)

        meta = result['SPY'].get('hierarchical_tuning', {}).get('phi_prior', {})
        self.assertIn('class_stats', meta, "Metadata should include class_stats")
        class_stats = meta['class_stats']
        self.assertIn('index', class_stats)
        self.assertIn('small_cap', class_stats)
        self.assertAlmostEqual(class_stats['index']['median'], 0.90, places=2)
        self.assertAlmostEqual(class_stats['small_cap']['median'], 0.25, places=2)

    def test_small_class_falls_back_to_global(self):
        """Classes with < PHI_POOL_MIN_ASSETS fall back to global median."""
        from tuning.tune import apply_cross_asset_phi_pooling

        entries = []
        # 6 large-cap assets
        for i, sym in enumerate(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']):
            entries.append((sym, 0.80, 500))
        # Only 2 crypto (below PHI_POOL_MIN_ASSETS=5)
        entries.append(('BTC-USD', 0.70, 500))
        entries.append(('ETH-USD', 0.65, 500))

        cache = self._build_cache(entries)
        result = apply_cross_asset_phi_pooling(cache)

        meta = result['AAPL'].get('hierarchical_tuning', {}).get('phi_prior', {})
        class_stats = meta.get('class_stats', {})
        if 'crypto' in class_stats:
            # Crypto class stats should use global median (not enough assets)
            # Global median of [0.80*6, 0.70, 0.65] = 0.80
            crypto_median = class_stats['crypto']['median']
            self.assertAlmostEqual(crypto_median, 0.80, places=1,
                                   msg="Crypto class with <5 assets should use global median")

    def test_no_regression_few_assets(self):
        """With fewer than PHI_POOL_MIN_ASSETS total, use default prior."""
        from tuning.tune import apply_cross_asset_phi_pooling, DEFAULT_PHI_PRIOR

        entries = [('SPY', 0.92, 500), ('AAPL', 0.85, 500)]
        cache = self._build_cache(entries)
        result = apply_cross_asset_phi_pooling(cache)

        meta = result['SPY'].get('hierarchical_tuning', {}).get('phi_prior', {})
        self.assertEqual(meta.get('phi_population_median'), DEFAULT_PHI_PRIOR)


class TestComputePhiPriorConstants(unittest.TestCase):
    """Verify that phi prior constants are sensible."""

    def test_constants_imported(self):
        from models.phi_student_t_unified import (
            PHI_PRIOR_INDEX,
            PHI_PRIOR_LARGE_CAP,
            PHI_PRIOR_SMALL_CAP,
            PHI_PRIOR_CRYPTO,
            PHI_PRIOR_METALS,
            PHI_PRIOR_FOREX,
            PHI_PRIOR_HIGH_VOL,
            PHI_PRIOR_DEFAULT,
            PHI_PRIOR_LAMBDA_BASE,
        )
        # Indices have highest momentum persistence
        self.assertEqual(PHI_PRIOR_INDEX, 0.95)
        # High-vol has lowest (strong mean reversion)
        self.assertEqual(PHI_PRIOR_HIGH_VOL, 0.20)
        # Lambda base is small but positive
        self.assertGreater(PHI_PRIOR_LAMBDA_BASE, 0)
        self.assertLess(PHI_PRIOR_LAMBDA_BASE, 1.0)

    def test_ordering(self):
        """Prior values should follow: high_vol < small_cap < forex <= default < crypto < large_cap < metals < index."""
        from models.phi_student_t_unified import (
            PHI_PRIOR_INDEX,
            PHI_PRIOR_LARGE_CAP,
            PHI_PRIOR_SMALL_CAP,
            PHI_PRIOR_CRYPTO,
            PHI_PRIOR_METALS,
            PHI_PRIOR_FOREX,
            PHI_PRIOR_HIGH_VOL,
            PHI_PRIOR_DEFAULT,
        )
        self.assertLess(PHI_PRIOR_HIGH_VOL, PHI_PRIOR_SMALL_CAP)
        self.assertLess(PHI_PRIOR_SMALL_CAP, PHI_PRIOR_FOREX)
        self.assertLessEqual(PHI_PRIOR_FOREX, PHI_PRIOR_DEFAULT)
        self.assertLess(PHI_PRIOR_DEFAULT, PHI_PRIOR_CRYPTO)
        self.assertLess(PHI_PRIOR_CRYPTO, PHI_PRIOR_LARGE_CAP)
        self.assertLess(PHI_PRIOR_LARGE_CAP, PHI_PRIOR_METALS)
        self.assertLessEqual(PHI_PRIOR_METALS, PHI_PRIOR_INDEX)


if __name__ == '__main__':
    unittest.main()
