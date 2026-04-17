"""
Tests for tune_modules/utilities.py -- Utility functions and model classification.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np


class TestIsQuietAndLog:
    """Verify _is_quiet and _log respect TUNING_QUIET."""

    def test_is_quiet_default_false(self, monkeypatch):
        monkeypatch.delenv('TUNING_QUIET', raising=False)
        from tuning.tune_modules.utilities import _is_quiet
        assert _is_quiet() is False

    def test_is_quiet_true_when_set(self, monkeypatch):
        monkeypatch.setenv('TUNING_QUIET', '1')
        from tuning.tune_modules.utilities import _is_quiet
        assert _is_quiet() is True

    def test_is_quiet_true_variants(self, monkeypatch):
        from tuning.tune_modules.utilities import _is_quiet
        for val in ('1', 'true', 'True', 'TRUE', 'yes', 'YES'):
            monkeypatch.setenv('TUNING_QUIET', val)
            assert _is_quiet() is True, f"TUNING_QUIET={val} should be quiet"

    def test_is_quiet_false_variants(self, monkeypatch):
        from tuning.tune_modules.utilities import _is_quiet
        for val in ('0', 'false', 'no', ''):
            monkeypatch.setenv('TUNING_QUIET', val)
            assert _is_quiet() is False, f"TUNING_QUIET={val} should not be quiet"

    def test_log_prints_when_not_quiet(self, monkeypatch, capsys):
        monkeypatch.delenv('TUNING_QUIET', raising=False)
        from tuning.tune_modules.utilities import _log
        _log("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_log_silent_when_quiet(self, monkeypatch, capsys):
        monkeypatch.setenv('TUNING_QUIET', '1')
        from tuning.tune_modules.utilities import _log
        _log("should not appear")
        captured = capsys.readouterr()
        assert captured.out == ""


class TestModelClass:
    """Verify ModelClass enum."""

    def test_model_class_values(self):
        from tuning.tune_modules.utilities import ModelClass
        assert ModelClass.KALMAN_GAUSSIAN == 0
        assert ModelClass.PHI_GAUSSIAN == 1
        assert ModelClass.PHI_STUDENT_T == 2

    def test_model_class_labels_exist(self):
        from tuning.tune_modules.utilities import ModelClass, MODEL_CLASS_LABELS
        for mc in ModelClass:
            assert mc in MODEL_CLASS_LABELS

    def test_model_class_n_params(self):
        from tuning.tune_modules.utilities import ModelClass, MODEL_CLASS_N_PARAMS
        assert MODEL_CLASS_N_PARAMS[ModelClass.KALMAN_GAUSSIAN] == 2
        assert MODEL_CLASS_N_PARAMS[ModelClass.PHI_GAUSSIAN] == 3
        assert MODEL_CLASS_N_PARAMS[ModelClass.PHI_STUDENT_T] == 3


class TestIsStudentTModel:
    """Comprehensive tests for is_student_t_model (100% branch coverage)."""

    @pytest.mark.parametrize("name,expected", [
        ("phi_student_t_nu_4", True),
        ("phi_student_t_nu_8", True),
        ("phi_student_t_nu_8_momentum", True),
        ("phi_student_t_nu_20", True),
        ("student_t", True),
        ("phi_student_t", True),
        ("PHI_STUDENT_T_NU_8", True),
        ("phi_student-t_nu_4", True),
        ("kalman_gaussian", False),
        ("kalman_phi_gaussian", False),
        ("gh_model", False),
        ("tvvm_model", False),
        ("", False),
    ])
    def test_is_student_t_model(self, name, expected):
        from tuning.tune_modules.utilities import is_student_t_model
        assert is_student_t_model(name) is expected

    def test_is_student_t_model_none_returns_false(self):
        from tuning.tune_modules.utilities import is_student_t_model
        assert is_student_t_model(None) is False


class TestIsHeavyTailedModel:
    """Comprehensive tests for is_heavy_tailed_model (100% branch coverage)."""

    @pytest.mark.parametrize("name,expected", [
        ("phi_student_t_nu_4", True),
        ("phi_student_t_nu_8", True),
        ("student_t", True),
        ("phi_student-t", True),
        ("phi_skew_t_nu_4_gamma_0.5", True),
        ("skew_t", True),
        ("kalman_gaussian", False),
        ("kalman_phi_gaussian", False),
        ("gh_model", False),
        ("", False),
    ])
    def test_is_heavy_tailed_model(self, name, expected):
        from tuning.tune_modules.utilities import is_heavy_tailed_model
        assert is_heavy_tailed_model(name) is expected

    def test_is_heavy_tailed_model_none_returns_false(self):
        from tuning.tune_modules.utilities import is_heavy_tailed_model
        assert is_heavy_tailed_model(None) is False


class TestComputeVolProportionalQFloor:
    """Tests for regime-conditional q floor computation."""

    def test_reference_vol_identity(self):
        """At reference vol, floor should equal base floor."""
        from tuning.tune_modules.utilities import (
            compute_vol_proportional_q_floor, Q_FLOOR_BY_REGIME, REFERENCE_VOL
        )
        for regime, q_base in Q_FLOOR_BY_REGIME.items():
            result = compute_vol_proportional_q_floor(regime, REFERENCE_VOL)
            assert abs(result - q_base) < 1e-12, f"regime {regime}: expected {q_base}, got {result}"

    def test_double_vol_quadruples_floor(self):
        """At 2x vol, floor should be 4x base."""
        from tuning.tune_modules.utilities import (
            compute_vol_proportional_q_floor, Q_FLOOR_BY_REGIME, REFERENCE_VOL
        )
        for regime, q_base in Q_FLOOR_BY_REGIME.items():
            result = compute_vol_proportional_q_floor(regime, 2 * REFERENCE_VOL)
            assert abs(result - 4.0 * q_base) < 1e-12

    def test_half_vol_quarters_floor(self):
        """At 0.5x vol, floor should be 0.25x base."""
        from tuning.tune_modules.utilities import (
            compute_vol_proportional_q_floor, Q_FLOOR_BY_REGIME, REFERENCE_VOL,
            Q_FLOOR_ABS_MIN
        )
        for regime, q_base in Q_FLOOR_BY_REGIME.items():
            result = compute_vol_proportional_q_floor(regime, 0.5 * REFERENCE_VOL)
            expected = max(0.25 * q_base, Q_FLOOR_ABS_MIN)
            assert abs(result - expected) < 1e-12

    def test_zero_vol_returns_abs_min(self):
        from tuning.tune_modules.utilities import compute_vol_proportional_q_floor, Q_FLOOR_ABS_MIN
        result = compute_vol_proportional_q_floor(0, 0.0)
        assert result >= Q_FLOOR_ABS_MIN

    def test_unknown_regime_returns_abs_min(self):
        from tuning.tune_modules.utilities import compute_vol_proportional_q_floor, Q_FLOOR_ABS_MIN
        result = compute_vol_proportional_q_floor(99, 0.16)
        assert result == Q_FLOOR_ABS_MIN

    def test_all_five_regimes_exist(self):
        from tuning.tune_modules.utilities import Q_FLOOR_BY_REGIME
        assert set(Q_FLOOR_BY_REGIME.keys()) == {0, 1, 2, 3, 4}


class TestConstants:
    """Verify key constants are present and have expected types/values."""

    def test_default_temporal_alpha(self):
        from tuning.tune_modules.utilities import DEFAULT_TEMPORAL_ALPHA
        assert DEFAULT_TEMPORAL_ALPHA == 0.3

    def test_min_hyvarinen_samples(self):
        from tuning.tune_modules.utilities import MIN_HYVARINEN_SAMPLES
        assert MIN_HYVARINEN_SAMPLES == 100

    def test_default_phi_prior(self):
        from tuning.tune_modules.utilities import DEFAULT_PHI_PRIOR
        assert DEFAULT_PHI_PRIOR == 0.85

    def test_phi_pool_min_assets(self):
        from tuning.tune_modules.utilities import PHI_POOL_MIN_ASSETS
        assert PHI_POOL_MIN_ASSETS == 5

    def test_phi_acf_scale(self):
        from tuning.tune_modules.utilities import PHI_ACF_SCALE
        assert PHI_ACF_SCALE == 2.0

    def test_q_floor_bic_lambda(self):
        from tuning.tune_modules.utilities import Q_FLOOR_BIC_LAMBDA
        assert Q_FLOOR_BIC_LAMBDA == 2.0

    def test_regime_labels_importable(self):
        from tuning.tune_modules.utilities import REGIME_LABELS
        assert isinstance(REGIME_LABELS, dict)

    def test_market_regime_importable(self):
        from tuning.tune_modules.utilities import MarketRegime
        assert MarketRegime is not None
