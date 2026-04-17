"""
Tests for tune_modules/config.py -- Feature flags and conditional imports.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


class TestFeatureFlags:
    """Verify all feature flags are boolean and have expected values."""

    def test_momentum_augmentation_is_bool(self):
        from tuning.tune_modules.config import MOMENTUM_AUGMENTATION_ENABLED
        assert isinstance(MOMENTUM_AUGMENTATION_ENABLED, bool)
        assert MOMENTUM_AUGMENTATION_ENABLED is True

    def test_unified_student_t_only_is_bool(self):
        from tuning.tune_modules.config import UNIFIED_STUDENT_T_ONLY
        assert isinstance(UNIFIED_STUDENT_T_ONLY, bool)
        assert UNIFIED_STUDENT_T_ONLY is True

    def test_gas_q_enabled_is_bool(self):
        from tuning.tune_modules.config import GAS_Q_ENABLED
        assert isinstance(GAS_Q_ENABLED, bool)

    def test_isotonic_recalibration_enabled_is_bool(self):
        from tuning.tune_modules.utilities import ISOTONIC_RECALIBRATION_ENABLED
        assert isinstance(ISOTONIC_RECALIBRATION_ENABLED, bool)

    def test_market_conditioning_enabled_is_bool(self):
        from tuning.tune_modules.config import MARKET_CONDITIONING_ENABLED
        assert isinstance(MARKET_CONDITIONING_ENABLED, bool)

    def test_elite_tuning_enabled_is_bool(self):
        from tuning.tune_modules.config import ELITE_TUNING_ENABLED
        assert isinstance(ELITE_TUNING_ENABLED, bool)

    def test_tvvm_enabled_flag_is_bool(self):
        from tuning.tune_modules.utilities import TVVM_ENABLED
        assert isinstance(TVVM_ENABLED, bool)

    def test_gh_model_enabled_is_bool(self):
        from tuning.tune_modules.utilities import GH_MODEL_ENABLED
        assert isinstance(GH_MODEL_ENABLED, bool)

    def test_adaptive_nu_enabled_is_bool(self):
        from tuning.tune_modules.utilities import ADAPTIVE_NU_ENABLED
        assert isinstance(ADAPTIVE_NU_ENABLED, bool)


class TestConditionalImports:
    """Verify availability flags from try/except import blocks."""

    def test_gas_q_availability_flag_exists(self):
        from tuning.tune_modules.config import GAS_Q_AVAILABLE
        assert isinstance(GAS_Q_AVAILABLE, bool)

    def test_gh_model_availability_flag_exists(self):
        from tuning.tune_modules.config import GH_MODEL_AVAILABLE
        assert isinstance(GH_MODEL_AVAILABLE, bool)

    def test_tvvm_availability_flag_exists(self):
        from tuning.tune_modules.config import TVVM_AVAILABLE
        assert isinstance(TVVM_AVAILABLE, bool)

    def test_isotonic_availability_flag_exists(self):
        from tuning.tune_modules.config import ISOTONIC_RECALIBRATION_AVAILABLE
        assert isinstance(ISOTONIC_RECALIBRATION_AVAILABLE, bool)

    def test_adaptive_nu_availability_flag_exists(self):
        from tuning.tune_modules.config import ADAPTIVE_NU_AVAILABLE
        assert isinstance(ADAPTIVE_NU_AVAILABLE, bool)

    def test_kahan_sum_availability_flag_exists(self):
        from tuning.tune_modules.config import KAHAN_SUM_AVAILABLE
        assert isinstance(KAHAN_SUM_AVAILABLE, bool)


class TestDataclasses:
    """Verify GASQFitResult is importable and works correctly."""

    def test_gasq_fit_result_importable(self):
        from tuning.tune_modules.config import GASQFitResult
        assert GASQFitResult is not None

    def test_gasq_fit_result_is_dataclass(self):
        from tuning.tune_modules.config import GASQFitResult
        import dataclasses
        assert dataclasses.is_dataclass(GASQFitResult)


class TestBackwardCompatibility:
    """Verify symbols can still be imported from tuning.tune."""

    def test_import_flag_from_tune(self):
        from tuning.tune import MOMENTUM_AUGMENTATION_ENABLED
        assert isinstance(MOMENTUM_AUGMENTATION_ENABLED, bool)

    def test_import_gasq_from_tune(self):
        from tuning.tune import GASQFitResult
        assert GASQFitResult is not None

    def test_import_is_student_t_from_tune(self):
        from tuning.tune import is_student_t_model
        assert callable(is_student_t_model)
