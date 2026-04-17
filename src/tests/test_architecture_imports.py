"""
Story 9.5 -- Import compatibility tests.

Verifies that every public symbol that was importable from ``tuning.tune``
and ``decision.signals`` is still importable from both the shim path and
the direct submodule path.

No computation; import-only.  Should run in < 10 seconds.
"""
from __future__ import annotations

import importlib
import os
import sys
import types

import pytest

# Ensure src/ is importable
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _assert_importable(module_path: str, name: str) -> None:
    """Assert *name* is importable from *module_path*."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, name), (
        f"{module_path} has no attribute {name!r}"
    )


def _assert_same_object(path_a: str, path_b: str, name: str) -> None:
    """Assert *name* resolves to the exact same object from both paths."""
    mod_a = importlib.import_module(path_a)
    mod_b = importlib.import_module(path_b)
    obj_a = getattr(mod_a, name)
    obj_b = getattr(mod_b, name)
    # For functions/classes, identity check is the gold standard.
    # For constants (int, float, str, tuple, dict) we fall back to equality.
    if isinstance(obj_a, (types.FunctionType, type)):
        assert obj_a is obj_b, (
            f"{name}: {path_a} -> {id(obj_a)} vs {path_b} -> {id(obj_b)}"
        )
    else:
        assert obj_a == obj_b, (
            f"{name}: {path_a} -> {obj_a!r} vs {path_b} -> {obj_b!r}"
        )


# =========================================================================
# Tune symbols
# =========================================================================

# -- Functions (shim: tuning.tune, direct: tuning.tune_modules) --
_TUNE_FUNCTIONS = [
    "tune_asset_with_bma",
    "main",
    "fit_all_models_for_regime",
    "kalman_filter_drift",
    "kalman_filter_drift_phi",
    "kalman_filter_drift_phi_student_t",
    "compute_pit_calibration_metrics",
    "run_calibration_pipeline",
    "load_cache",
    "save_cache_json",
    "needs_retune",
    "fit_garch_mle",
    "tune_asset_q",
    "entropy_regularized_bma",
    "compute_regime_diagnostics",
    "assign_regime_labels",
    "compute_combined_model_weights",
    "validate_tune_result",
    "stamp_tune_result",
    "sort_assets_by_complexity",
    "load_asset_list",
    "compute_crps_gaussian_inline",
    "compute_crps_student_t_inline",
    "compute_extended_pit_metrics_gaussian",
    "compute_extended_pit_metrics_student_t",
    "compute_composite_volatility",
    "detect_overnight_gap",
    "innovation_cusum",
    "rolling_phi_estimate",
    "fit_gas_q_gaussian",
    "fit_gas_q_student_t",
    "vectorized_bma_weights",
    "apply_pit_penalties_to_weights",
    "tune_regime_model_averaging",
    "compute_hyvarinen_score_gaussian",
    "compute_hyvarinen_score_student_t",
]

_TUNE_CLASSES = [
    "GaussianDriftModel",
    "PhiStudentTDriftModel",
    "MomentumConfig",
    "EliteTuningConfig",
    "HMMRegimeResult",
    "FilterCacheKey",
    "FilterCacheValue",
    "TuningAuditRecord",
    "MarketRegime",
    "GASQConfig",
    "GASQFitResult",
    "GASQResult",
    "MomentumAugmentedDriftModel",
    "PhiGaussianDriftModel",
    "AdaptiveNuConfig",
    "RVAdaptiveQConfig",
    "RVAdaptiveQResult",
    "ModelFamily",
    "EliteTuningDiagnostics",
    "GHModelConfig",
]

_TUNE_CONSTANTS = [
    "FILTER_CACHE_AVAILABLE",
    "FILTER_CACHE_ENABLED",
    "GAS_Q_AVAILABLE",
    "GAS_Q_ENABLED",
    "GARCH_ALPHA_MIN",
    "GARCH_ALPHA_MAX",
    "GARCH_BETA_MIN",
    "GARCH_BETA_MAX",
    "DEFAULT_GAS_Q_CONFIG",
    "DEFAULT_MOMENTUM_CONFIG",
    "ELITE_TUNING_AVAILABLE",
    "ELITE_TUNING_ENABLED",
    "DEFAULT_REFINEMENT_CONFIG",
    "GH_MODEL_AVAILABLE",
    "DEFAULT_MODEL_SELECTION_METHOD",
]


class TestTuneShimImports:
    """Every symbol importable from tuning.tune must also exist in tuning.tune_modules."""

    @pytest.mark.parametrize("name", _TUNE_FUNCTIONS)
    def test_tune_function_importable_from_shim(self, name: str) -> None:
        _assert_importable("tuning.tune", name)

    @pytest.mark.parametrize("name", _TUNE_FUNCTIONS)
    def test_tune_function_importable_from_modules(self, name: str) -> None:
        _assert_importable("tuning.tune_modules", name)

    @pytest.mark.parametrize("name", _TUNE_FUNCTIONS)
    def test_tune_function_same_object(self, name: str) -> None:
        _assert_same_object("tuning.tune", "tuning.tune_modules", name)

    @pytest.mark.parametrize("name", _TUNE_CLASSES)
    def test_tune_class_importable_from_shim(self, name: str) -> None:
        _assert_importable("tuning.tune", name)

    @pytest.mark.parametrize("name", _TUNE_CLASSES)
    def test_tune_class_importable_from_modules(self, name: str) -> None:
        _assert_importable("tuning.tune_modules", name)

    @pytest.mark.parametrize("name", _TUNE_CLASSES)
    def test_tune_class_same_object(self, name: str) -> None:
        _assert_same_object("tuning.tune", "tuning.tune_modules", name)

    @pytest.mark.parametrize("name", _TUNE_CONSTANTS)
    def test_tune_constant_importable_from_shim(self, name: str) -> None:
        _assert_importable("tuning.tune", name)

    @pytest.mark.parametrize("name", _TUNE_CONSTANTS)
    def test_tune_constant_importable_from_modules(self, name: str) -> None:
        _assert_importable("tuning.tune_modules", name)

    @pytest.mark.parametrize("name", _TUNE_CONSTANTS)
    def test_tune_constant_same_value(self, name: str) -> None:
        _assert_same_object("tuning.tune", "tuning.tune_modules", name)


class TestTuneDirectSubmoduleImports:
    """Key symbols are importable directly from their source submodule."""

    _DIRECT_IMPORTS = [
        ("tuning.tune_modules.asset_tuning", "tune_asset_with_bma"),
        ("tuning.tune_modules.cli", "main"),
        ("tuning.tune_modules.model_fitting", "fit_all_models_for_regime"),
        ("tuning.tune_modules.kalman_wrappers", "kalman_filter_drift"),
        ("tuning.tune_modules.kalman_wrappers", "kalman_filter_drift_phi"),
        ("tuning.tune_modules.kalman_wrappers", "kalman_filter_drift_phi_student_t"),
        ("tuning.tune_modules.volatility_fitting", "fit_garch_mle"),
        ("tuning.tune_modules.volatility_fitting", "load_cache"),
        ("tuning.tune_modules.volatility_fitting", "save_cache_json"),
        ("tuning.tune_modules.process_noise", "needs_retune"),
        ("tuning.tune_modules.process_noise", "tune_asset_q"),
        ("tuning.tune_modules.calibration_pipeline", "run_calibration_pipeline"),
        ("tuning.tune_modules.regime_bma", "tune_regime_model_averaging"),
        ("tuning.tune_modules.regime_bma", "vectorized_bma_weights"),
        ("tuning.tune_modules.config", "FILTER_CACHE_AVAILABLE"),
        ("tuning.tune_modules.process_noise", "sort_assets_by_complexity"),
    ]

    @pytest.mark.parametrize("module_path,name", _DIRECT_IMPORTS)
    def test_direct_import(self, module_path: str, name: str) -> None:
        _assert_importable(module_path, name)


# =========================================================================
# Signal symbols
# =========================================================================

_SIGNAL_FUNCTIONS = [
    "process_single_asset",
    "latest_signals",
    "main",
    "parse_args",
    "compute_features",
    "bayesian_model_average_mc",
    "run_unified_mc",
    "label_from_probability",
    "compute_all_diagnostics",
    "compute_pnl_attribution",
    "save_signal_state",
    "load_signal_state",
    "fit_hmm_regimes",
    "garch_variance_forecast",
    "compute_features",
    "sign_prob_with_uncertainty",
    "multi_horizon_sign_prob",
    "dynamic_leverage",
    "kelly_fraction",
    "drawdown_adjusted_kelly",
    "compute_conviction",
    "adaptive_momentum_weights",
    "compute_risk_temperature",
    "compute_adaptive_thresholds",
    "compute_evt_var",
    "compute_evt_expected_loss",
    "cross_asset_confirmation",
    "run_walk_forward_backtest",
    "isotonic_recalibrate",
    "decay_signal",
    "normalize_portfolio",
    "compute_unified_risk_context",
    "rank_by_conviction",
    "render_detailed_signal_table",
    "render_simplified_signal_table",
    "render_multi_asset_summary_table",
    "render_strong_signals_summary",
    "download_prices_bulk",
    "compute_directional_exhaustion_from_features",
    "vec_bma_weights",
]

_SIGNAL_CLASSES = [
    "Signal",
    "GASQConfig",
    "GASQResult",
    "CUSUMParams",
    "WalkForwardResult",
    "IsotonicRecalibrator",
    "VolatilityEstimator",
    "AdaptiveThresholds",
    "RiskTemperatureResult",
    "UnifiedRiskContext",
    "CrossAssetConfirmation",
    "MCDiagnostics",
    "EscalationDecision",
    "PITViolationResult",
    "PITPenaltyReport",
    "ModelFamily",
    "MRSignalResult",
    "CalibrationDiagnostics",
    "ComputationCache",
    "CalibratedTrust",
]

_SIGNAL_CONSTANTS = [
    "DEFAULT_HORIZONS",
    "EDGE_FLOOR",
    "EDGE_FLOOR_Z",
    "DEFAULT_CACHE_PATH",
    "HMM_AVAILABLE",
    "GAS_Q_AVAILABLE",
    "FILTER_CACHE_AVAILABLE",
    "CRPS_AVAILABLE",
    "EVT_AVAILABLE",
    "DISPLAY_PRICE_INERTIA",
    "RETURN_CAP_DEFAULT",
    "MOM_DRIFT_SCALE",
    "MOM_CROSSOVER_HORIZON",
    "MOMENTUM_HALF_LIFE_DAYS",
]


class TestSignalShimImports:
    """Every symbol importable from decision.signals must also exist in decision.signal_modules."""

    @pytest.mark.parametrize("name", _SIGNAL_FUNCTIONS)
    def test_signal_function_importable_from_shim(self, name: str) -> None:
        _assert_importable("decision.signals", name)

    @pytest.mark.parametrize("name", _SIGNAL_FUNCTIONS)
    def test_signal_function_importable_from_modules(self, name: str) -> None:
        _assert_importable("decision.signal_modules", name)

    @pytest.mark.parametrize("name", _SIGNAL_FUNCTIONS)
    def test_signal_function_same_object(self, name: str) -> None:
        _assert_same_object("decision.signals", "decision.signal_modules", name)

    @pytest.mark.parametrize("name", _SIGNAL_CLASSES)
    def test_signal_class_importable_from_shim(self, name: str) -> None:
        _assert_importable("decision.signals", name)

    @pytest.mark.parametrize("name", _SIGNAL_CLASSES)
    def test_signal_class_importable_from_modules(self, name: str) -> None:
        _assert_importable("decision.signal_modules", name)

    @pytest.mark.parametrize("name", _SIGNAL_CLASSES)
    def test_signal_class_same_object(self, name: str) -> None:
        _assert_same_object("decision.signals", "decision.signal_modules", name)

    @pytest.mark.parametrize("name", _SIGNAL_CONSTANTS)
    def test_signal_constant_importable_from_shim(self, name: str) -> None:
        _assert_importable("decision.signals", name)

    @pytest.mark.parametrize("name", _SIGNAL_CONSTANTS)
    def test_signal_constant_importable_from_modules(self, name: str) -> None:
        _assert_importable("decision.signal_modules", name)

    @pytest.mark.parametrize("name", _SIGNAL_CONSTANTS)
    def test_signal_constant_same_value(self, name: str) -> None:
        _assert_same_object("decision.signals", "decision.signal_modules", name)


class TestSignalDirectSubmoduleImports:
    """Key symbols are importable directly from their source submodule."""

    _DIRECT_IMPORTS = [
        ("decision.signal_modules.asset_processing", "process_single_asset"),
        ("decision.signal_modules.signal_generation", "latest_signals"),
        ("decision.signal_modules.cli", "main"),
        ("decision.signal_modules.cli", "parse_args"),
        ("decision.signal_modules.feature_pipeline", "compute_features"),
        ("decision.signal_modules.bma_engine", "bayesian_model_average_mc"),
        ("decision.signal_modules.monte_carlo", "run_unified_mc"),
        ("decision.signal_modules.probability_mapping", "label_from_probability"),
        ("decision.signal_modules.comprehensive_diagnostics", "compute_all_diagnostics"),
        ("decision.signal_modules.pnl_attribution", "compute_pnl_attribution"),
        ("decision.signal_modules.signal_state", "save_signal_state"),
        ("decision.signal_modules.signal_state", "load_signal_state"),
        ("decision.signal_modules.hmm_regimes", "fit_hmm_regimes"),
        ("decision.signal_modules.signal_dataclass", "Signal"),
        ("decision.signal_modules.walk_forward", "WalkForwardResult"),
        ("decision.signal_modules.config", "HMM_AVAILABLE"),
        ("decision.signal_modules.signal_generation", "compute_adaptive_thresholds"),
    ]

    @pytest.mark.parametrize("module_path,name", _DIRECT_IMPORTS)
    def test_direct_import(self, module_path: str, name: str) -> None:
        _assert_importable(module_path, name)


# =========================================================================
# Private symbol re-exports (shim must re-export these explicitly)
# =========================================================================

class TestPrivateSymbolReexports:
    """Private symbols that the shim explicitly re-exports."""

    _SIGNAL_PRIVATES = [
        ("decision.signals", "_process_assets_with_retries"),
        ("decision.signals", "_to_float"),
        ("decision.signals", "_download_prices"),
        ("decision.signals", "_resolve_display_name"),
        ("decision.signals", "_fetch_px_symbol"),
        ("decision.signals", "_fetch_with_fallback"),
        ("decision.signals", "_as_series"),
        ("decision.signals", "_ensure_float_series"),
        ("decision.signals", "_align_fx_asof"),
        ("decision.signals", "_resolve_symbol_candidates"),
        ("decision.signals", "_enrich_signal_with_epic8"),
        ("decision.signals", "_SIG_H_ANNUAL_CAP"),
        ("decision.signals", "_CI_LOG_FLOOR"),
        ("decision.signals", "_CI_LOG_CAP"),
        ("decision.signals", "_DISPLAY_PRICE_CACHE"),
        ("decision.signals", "_compute_sig_h_cap"),
        ("decision.signals", "_smooth_display_price"),
        ("decision.signals", "_logistic"),
        ("decision.signals", "_CUSUM_STATE"),
        ("decision.signals", "_get_cusum_state"),
        ("decision.signals", "_compute_simple_exhaustion"),
        ("decision.signals", "_garch11_mle"),
        ("decision.signals", "_fit_student_nu_mle"),
        ("decision.signals", "_test_innovation_whiteness"),
        ("decision.signals", "_compute_kalman_log_likelihood"),
        ("decision.signals", "_compute_kalman_log_likelihood_heteroskedastic"),
        ("decision.signals", "_estimate_regime_drift_priors"),
        ("decision.signals", "_safe_get_nested"),
        ("decision.signals", "_load_tuned_kalman_params"),
        ("decision.signals", "_select_regime_params"),
        ("decision.signals", "_compute_kalman_gain_from_filtered"),
        ("decision.signals", "_apply_gain_monitoring_reset"),
        ("decision.signals", "_kalman_filter_drift"),
        ("decision.signals", "_WFFeatureCache"),
        ("decision.signals", "_simulate_forward_paths"),
        ("decision.signals", "_load_signals_calibration"),
        ("decision.signals", "_apply_single_p_map"),
        ("decision.signals", "_apply_p_up_calibration"),
        ("decision.signals", "_apply_emos_correction"),
        ("decision.signals", "_apply_magnitude_bias_correction"),
        ("decision.signals", "_get_calibrated_label_thresholds"),
    ]

    @pytest.mark.parametrize("module_path,name", _SIGNAL_PRIVATES)
    def test_signal_private_importable(self, module_path: str, name: str) -> None:
        _assert_importable(module_path, name)


# =========================================================================
# Module-level constants defined in the shim itself
# =========================================================================

class TestShimOnlyConstants:
    """Constants defined directly in the shim files (not in submodules)."""

    _SIGNAL_SHIM_CONSTANTS = [
        "PAIR",
        "DEFAULT_HORIZONS",
        "NOTIONAL_PLN",
        "EDGE_FLOOR",
        "EDGE_FLOOR_Z",
        "DEFAULT_CACHE_PATH",
        "MOM_DRIFT_SCALE",
        "MOM_CROSSOVER_HORIZON",
        "MOM_MIN_OBSERVATIONS",
        "MOMENTUM_HALF_LIFE_DAYS",
        "MOM_SLOW_FRAC",
        "SLOW_Q_RATIO",
        "COHERENT_MC_ENABLED",
        "QUANTILE_CI_MIN_SAMPLES",
        "RETURN_CAP_BY_CLASS",
        "RETURN_CAP_DEFAULT",
    ]

    @pytest.mark.parametrize("name", _SIGNAL_SHIM_CONSTANTS)
    def test_signal_shim_constant(self, name: str) -> None:
        _assert_importable("decision.signals", name)


# =========================================================================
# Smoke test: module imports do not raise
# =========================================================================

class TestModuleImportSmoke:
    """Importing any submodule must not raise."""

    _TUNE_SUBMODULES = [
        "tuning.tune_modules.config",
        "tuning.tune_modules.utilities",
        "tuning.tune_modules.calibration_pipeline",
        "tuning.tune_modules.process_noise",
        "tuning.tune_modules.volatility_fitting",
        "tuning.tune_modules.kalman_wrappers",
        "tuning.tune_modules.pit_diagnostics",
        "tuning.tune_modules.model_fitting",
        "tuning.tune_modules.regime_bma",
        "tuning.tune_modules.asset_tuning",
        "tuning.tune_modules.cli",
    ]
    _SIGNAL_SUBMODULES = [
        "decision.signal_modules.config",
        "decision.signal_modules.volatility_imports",
        "decision.signal_modules.regime_classification",
        "decision.signal_modules.data_fetching",
        "decision.signal_modules.momentum_features",
        "decision.signal_modules.kalman_diagnostics",
        "decision.signal_modules.parameter_loading",
        "decision.signal_modules.kalman_filtering",
        "decision.signal_modules.hmm_regimes",
        "decision.signal_modules.feature_pipeline",
        "decision.signal_modules.monte_carlo",
        "decision.signal_modules.bma_engine",
        "decision.signal_modules.walk_forward",
        "decision.signal_modules.signal_dataclass",
        "decision.signal_modules.threshold_calibration",
        "decision.signal_modules.probability_mapping",
        "decision.signal_modules.signal_generation",
        "decision.signal_modules.signal_state",
        "decision.signal_modules.pnl_attribution",
        "decision.signal_modules.comprehensive_diagnostics",
        "decision.signal_modules.asset_processing",
        "decision.signal_modules.cli",
    ]

    @pytest.mark.parametrize("module_path", _TUNE_SUBMODULES + _SIGNAL_SUBMODULES)
    def test_submodule_imports_without_error(self, module_path: str) -> None:
        mod = importlib.import_module(module_path)
        assert mod is not None
