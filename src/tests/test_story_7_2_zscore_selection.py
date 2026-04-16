"""
Story 7.2: EWM vs Expanding Window Z-Score Selection
=====================================================
Per-asset BIC-driven selection of z-score baseline for MS-q.
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
    get_ms_z_method,
    select_ms_z_method_auto,
    MS_Z_ASSET_CLASS_DEFAULTS,
)


class TestEWMvsExpandingZScore:
    """Acceptance criteria for Story 7.2."""

    def test_metals_prefer_ewm_097(self):
        """AC1: Metals prefer EWM with lambda=0.97."""
        method, lam = get_ms_z_method('GC=F')
        assert method == 'ewm' and lam == 0.97

    def test_high_vol_prefer_ewm_094(self):
        """AC2: High vol equity prefers EWM with lambda=0.94."""
        method, lam = get_ms_z_method('MSTR')
        assert method == 'ewm' and lam == 0.94

    def test_index_prefer_expanding(self):
        """AC3: Indices prefer expanding window."""
        method, lam = get_ms_z_method('SPY')
        assert method == 'expanding' and lam == 0.0

    def test_auto_selection_returns_valid(self):
        """AC4: Auto selection returns valid method and diagnostics."""
        np.random.seed(42)
        vol = np.abs(np.random.normal(0.02, 0.005, 500))
        method, lam, diag = select_ms_z_method_auto(
            vol, q_calm=1e-6, q_stress=1e-4
        )
        assert method in ('ewm', 'expanding')
        assert 'expanding' in diag
        assert any(k.startswith('ewm_') for k in diag)

    def test_convergence_short_series(self):
        """AC5: Expanding window requires n > 100; auto switches to EWM for short series."""
        np.random.seed(99)
        vol = np.abs(np.random.normal(0.02, 0.005, 50))
        method, lam, _ = select_ms_z_method_auto(
            vol, q_calm=1e-6, q_stress=1e-4
        )
        # Short series should not use expanding
        if method == 'expanding':
            pytest.fail("Expected EWM for short series n=50")
        assert lam > 0

    def test_bic_with_returns(self):
        """AC6: Auto selection uses returns when provided."""
        np.random.seed(42)
        n = 300
        vol = np.abs(np.random.normal(0.02, 0.005, n))
        returns = np.random.normal(0, 0.015, n)
        method, lam, diag = select_ms_z_method_auto(
            vol, q_calm=1e-6, q_stress=1e-4, returns=returns
        )
        assert method in ('ewm', 'expanding')
        assert len(diag) == 4  # 3 EWM + 1 expanding

    def test_all_asset_classes_defined(self):
        """AC7: All expected asset classes have z-score defaults."""
        for cls in ['metals_gold', 'metals_silver', 'high_vol_equity', 'crypto', 'index']:
            assert cls in MS_Z_ASSET_CLASS_DEFAULTS, f"Missing {cls}"

    def test_crypto_prefers_ewm(self):
        """AC8: Crypto prefers EWM."""
        method, lam = get_ms_z_method('BTC-USD')
        assert method == 'ewm' and lam == 0.95
