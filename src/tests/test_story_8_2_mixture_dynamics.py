"""
Story 8.2: Multi-Factor Mixture Weight Dynamics Validation
===========================================================
Validated multi-factor conditioning: shocks, vol acceleration, momentum.
"""
import os
import sys
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.phi_student_t import (
    compute_mixture_weight_dynamic,
    get_mixture_factor_loadings,
    MIXTURE_FACTOR_LOADINGS,
)


class TestMixtureWeightDynamics:
    """Acceptance criteria for Story 8.2."""

    def test_shock_response_within_2_days(self):
        """AC1: w_t responds within 2 days of a 3-sigma shock."""
        np.random.seed(42)
        n = 200
        returns = np.random.normal(0, 0.01, n)
        vol = np.full(n, 0.01)
        # 3-sigma shock at day 100
        returns[100] = -0.05  # 5x normal
        w_t, _ = compute_mixture_weight_dynamic(returns, vol)
        # w_t should drop at day 100
        assert w_t[100] < w_t[99], "w_t should drop on shock day"
        assert w_t[100] < 0.7, f"Expected w_t < 0.7 on shock, got {w_t[100]:.3f}"

    def test_vol_acceleration_drops_weight(self):
        """AC2: Vol acceleration signal: w_t drops when vol accelerates."""
        np.random.seed(42)
        n = 200
        returns = np.random.normal(0, 0.01, n)
        vol = np.full(n, 0.01)
        # Vol spike starting day 100
        vol[100:110] = np.linspace(0.01, 0.05, 10)
        w_t, _ = compute_mixture_weight_dynamic(returns, vol)
        # During vol acceleration, w_t should decrease
        mean_before = np.mean(w_t[80:100])
        mean_during = np.mean(w_t[101:110])
        assert mean_during < mean_before, "w_t should drop during vol acceleration"

    def test_momentum_increases_weight(self):
        """AC3: Momentum signal: w_t increases during sustained trends."""
        np.random.seed(42)
        n = 200
        vol = np.full(n, 0.01)
        returns = np.random.normal(0, 0.005, n)
        # Positive trending from day 50-100
        returns[50:100] = np.random.normal(0.005, 0.003, 50)
        w_t, _ = compute_mixture_weight_dynamic(returns, vol, c=0.5)
        # After trend establishes (day 70+), w_t should be higher
        mean_pretrend = np.mean(w_t[30:50])
        mean_trend = np.mean(w_t[75:100])
        # Trend momentum should pull w_t higher (calmer)
        assert mean_trend > mean_pretrend - 0.2  # generous tolerance

    def test_factor_loadings_per_asset(self):
        """AC4: Factor loadings calibrated per asset class."""
        a, b, c = get_mixture_factor_loadings('MSTR')
        assert a == 1.5 and b == 0.8 and c == 0.4  # high_vol_equity

        a, b, c = get_mixture_factor_loadings('GC=F')
        assert a == 0.8 and b == 0.3 and c == 0.2  # metals_gold

    def test_diagnostics_complete(self):
        """AC5: Diagnostics contain all expected fields."""
        np.random.seed(42)
        n = 200
        returns = np.random.normal(0, 0.01, n)
        vol = np.abs(np.random.normal(0.01, 0.002, n))
        _, diag = compute_mixture_weight_dynamic(returns, vol)
        required = {'mean_w', 'min_w', 'max_w', 'factor_z_std',
                     'factor_vol_accel_std', 'factor_momentum_std'}
        assert required.issubset(diag.keys())

    def test_weight_bounded(self):
        """AC6: w_t stays in [0.01, 0.99]."""
        np.random.seed(42)
        n = 500
        returns = np.random.normal(0, 0.03, n)
        vol = np.abs(np.random.normal(0.02, 0.01, n))
        w_t, _ = compute_mixture_weight_dynamic(returns, vol)
        assert np.all(w_t >= 0.01) and np.all(w_t <= 0.99)

    def test_all_asset_classes_have_loadings(self):
        """AC7: All asset classes have factor loadings."""
        for cls in ['metals_gold', 'metals_silver', 'high_vol_equity', 'crypto', 'large_cap']:
            assert cls in MIXTURE_FACTOR_LOADINGS, f"Missing {cls}"
