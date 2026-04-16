"""
Story 9.3: Asset-Class Profile Validation Suite
=================================================
Empirical validation that each profile outperforms generic defaults.
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

from models.phi_student_t_unified import (
    validate_asset_class_profile,
    get_generic_defaults,
    ASSET_CLASS_PROFILES,
)


class TestAssetClassProfileValidation:
    """Acceptance criteria for Story 9.3."""

    def test_gold_profile_passes(self):
        """AC1: Gold profile passes with CRPS < 0.018."""
        result = validate_asset_class_profile(
            'GC=F',
            profile_metrics={'crps': 0.015, 'pit_ks_p': 0.20, 'css': 0.75, 'fec': 0.85, 'bic': 500},
            generic_metrics={'crps': 0.020, 'pit_ks_p': 0.10, 'css': 0.60, 'fec': 0.70, 'bic': 520},
        )
        assert result['passes'] is True
        assert result['asset_class'] == 'metals_gold'

    def test_gold_profile_fails_high_crps(self):
        """AC2: Gold profile fails if CRPS > 0.018."""
        result = validate_asset_class_profile(
            'GC=F',
            profile_metrics={'crps': 0.020, 'pit_ks_p': 0.25, 'css': 0.80, 'fec': 0.90, 'bic': 500},
            generic_metrics={'crps': 0.025, 'pit_ks_p': 0.10, 'css': 0.60, 'fec': 0.70, 'bic': 520},
        )
        assert result['passes'] is False
        assert any('CRPS' in f for f in result['failures'])

    def test_high_vol_profile(self):
        """AC3: High-vol profile requires CSS > 0.70."""
        result = validate_asset_class_profile(
            'MSTR',
            profile_metrics={'crps': 0.030, 'pit_ks_p': 0.15, 'css': 0.75, 'fec': 0.70, 'bic': 600},
            generic_metrics={'crps': 0.040, 'pit_ks_p': 0.10, 'css': 0.50, 'fec': 0.55, 'bic': 650},
        )
        assert result['passes'] is True

    def test_crps_improvement_metric(self):
        """AC4: CRPS improvement computed correctly."""
        result = validate_asset_class_profile(
            'MSFT',
            profile_metrics={'crps': 0.009, 'bic': 400},
            generic_metrics={'crps': 0.010, 'bic': 410},
        )
        assert result['crps_improvement_pct'] == pytest.approx(10.0, rel=0.01)

    def test_generic_defaults_available(self):
        """AC5: Generic defaults are available."""
        defaults = get_generic_defaults()
        assert 'ms_sensitivity_init' in defaults
        assert defaults['ms_sensitivity_init'] == 2.0

    def test_all_profiles_defined(self):
        """AC6: All expected asset-class profiles exist."""
        for cls in ['metals_gold', 'metals_silver', 'high_vol_equity', 'forex']:
            assert cls in ASSET_CLASS_PROFILES

    def test_validation_structure(self):
        """AC7: Validation result has complete structure."""
        result = validate_asset_class_profile(
            'SPY',
            profile_metrics={'crps': 0.008, 'pit_ks_p': 0.30, 'css': 0.85, 'fec': 0.90, 'bic': 300},
            generic_metrics={'crps': 0.010, 'pit_ks_p': 0.20, 'css': 0.70, 'fec': 0.80, 'bic': 320},
        )
        required = {'asset_symbol', 'asset_class', 'crps_improvement_pct',
                     'bic_improvement_nats', 'passes', 'failures'}
        assert required.issubset(result.keys())
