"""
Story 7.1: Calibrated MS-q Sensitivity per Asset Class
========================================================
Asset-class specific sigmoid parameters for regime detection.
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

from models.phi_student_t import get_ms_q_params, MS_Q_ASSET_CLASS_PARAMS


class TestCalibratedMSqSensitivity:
    """Acceptance criteria for Story 7.1."""

    def test_gold_params(self):
        """AC1: Gold: sensitivity=4.0, threshold=1.5."""
        sens, thresh = get_ms_q_params('GC=F')
        assert sens == 4.0 and thresh == 1.5

    def test_silver_params(self):
        """AC2: Silver: sensitivity=4.5, threshold=1.2."""
        sens, thresh = get_ms_q_params('SI=F')
        assert sens == 4.5 and thresh == 1.2

    def test_large_cap_params(self):
        """AC3: Large cap: sensitivity=2.5, threshold=1.3."""
        sens, thresh = get_ms_q_params('MSFT')
        assert sens == 2.5 and thresh == 1.3
        sens2, thresh2 = get_ms_q_params('GOOGL')
        assert sens2 == 2.5 and thresh2 == 1.3

    def test_high_vol_params(self):
        """AC4: High vol: sensitivity=3.0, threshold=1.0."""
        sens, thresh = get_ms_q_params('MSTR')
        assert sens == 3.0 and thresh == 1.0

    def test_crypto_params(self):
        """AC5: Crypto: sensitivity=3.5, threshold=1.1."""
        sens, thresh = get_ms_q_params('BTC-USD')
        assert sens == 3.5 and thresh == 1.1

    def test_transition_speed(self):
        """AC6: p_stress reaches 0.90 within 5 days of vol spike."""
        from models.phi_student_t import compute_ms_process_noise_smooth
        # Create vol series with spike at day 100
        n = 200
        vol = np.full(n, 0.02)
        vol[100:] = 0.08  # 4x vol spike

        for asset, (sens, thresh) in [('GC=F', (4.0, 1.5)), ('MSTR', (3.0, 1.0))]:
            _, p_stress = compute_ms_process_noise_smooth(
                vol, q_calm=1e-6, q_stress=1e-4,
                sensitivity=sens
            )
            # Within 5 days of spike (day 100-105), p_stress should be high
            max_p_105 = float(np.max(p_stress[100:106]))
            assert max_p_105 > 0.80, \
                f"{asset}: p_stress max in [100,105] = {max_p_105:.3f}, expected > 0.80"

    def test_default_for_unknown(self):
        """AC7: Unknown symbols get default (2.0, 1.3) params."""
        sens, thresh = get_ms_q_params('UNKNOWN_TICKER')
        assert sens == 2.0 and thresh == 1.3

    def test_all_classes_defined(self):
        """AC8: All expected asset classes have MS-q params."""
        for cls in ['metals_gold', 'metals_silver', 'high_vol_equity', 'crypto', 'index']:
            assert cls in MS_Q_ASSET_CLASS_PARAMS, f"Missing {cls}"
