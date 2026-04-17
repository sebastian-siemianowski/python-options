"""Comprehensive diagnostics for signal pipeline.

Extracted from signals.py - Story 8.5.
Contains compute_all_diagnostics() which aggregates log-likelihoods,
parameter stability, out-of-sample tests, PIT calibration, and model comparison.
"""

import os
from typing import Dict

import numpy as np
import pandas as pd

import sys as _sys
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

from decision.signal_modules.config import *  # noqa: F403
from decision.signal_modules.hmm_regimes import *  # noqa: F403
from decision.signal_modules.walk_forward import *  # noqa: F403


def compute_all_diagnostics(px: pd.Series, feats: Dict[str, pd.Series], enable_oos: bool = False, enable_pit_calibration: bool = False, enable_model_comparison: bool = False) -> Dict:
    """
    Compute comprehensive diagnostics: log-likelihood monitoring, parameter stability,
    and optionally out-of-sample tests, PIT calibration verification, and structural model comparison.

    Args:
        px: Price series
        feats: Feature dictionary from compute_features
        enable_oos: If True, run expensive out-of-sample validation
        enable_pit_calibration: If True, run PIT calibration verification (expensive)
        enable_model_comparison: If True, run structural model comparison (AIC/BIC falsifiability)

    Returns:
        Dictionary with all diagnostic metrics
    """
    diagnostics = {}

    # 1. Log-likelihood monitoring from fitted models
    garch_params = feats.get("garch_params", {})
    if isinstance(garch_params, dict):
        diagnostics["garch_log_likelihood"] = garch_params.get("log_likelihood", float("nan"))
        diagnostics["garch_aic"] = garch_params.get("aic", float("nan"))
        diagnostics["garch_bic"] = garch_params.get("bic", float("nan"))
        diagnostics["garch_n_obs"] = garch_params.get("n_obs", 0)

    # Pillar 1: Kalman filter drift diagnostics (with refinements)
    kalman_metadata = feats.get("kalman_metadata", {})
    if isinstance(kalman_metadata, dict):
        diagnostics["kalman_log_likelihood"] = kalman_metadata.get("log_likelihood", float("nan"))
        diagnostics["kalman_process_noise_var"] = kalman_metadata.get("process_noise_var", float("nan"))
        diagnostics["kalman_n_obs"] = kalman_metadata.get("n_obs", 0)
        # Refinement 1: q optimization results
        diagnostics["kalman_q_optimal"] = kalman_metadata.get("q_optimal", float("nan"))
        diagnostics["kalman_q_heuristic"] = kalman_metadata.get("q_heuristic", float("nan"))
        diagnostics["kalman_q_optimization_attempted"] = kalman_metadata.get("q_optimization_attempted", False)
        # Refinement 2: Kalman gain statistics (situational awareness)
        diagnostics["kalman_gain_mean"] = kalman_metadata.get("kalman_gain_mean", float("nan"))
        diagnostics["kalman_gain_recent"] = kalman_metadata.get("kalman_gain_recent", float("nan"))
        # Refinement 3: Innovation whiteness test (model adequacy)
        innovation_whiteness = kalman_metadata.get("innovation_whiteness", {})
        if isinstance(innovation_whiteness, dict):
            diagnostics["innovation_ljung_box_statistic"] = innovation_whiteness.get("ljung_box_statistic", float("nan"))
            diagnostics["innovation_ljung_box_pvalue"] = innovation_whiteness.get("ljung_box_pvalue", float("nan"))
            diagnostics["innovation_model_adequate"] = innovation_whiteness.get("model_adequate", None)
            diagnostics["innovation_lags_tested"] = innovation_whiteness.get("lags_tested", 0)
        # Level-7 Refinement: Heteroskedastic process noise (q_t = c * sigma_t^2)
        diagnostics["kalman_heteroskedastic_mode"] = kalman_metadata.get("heteroskedastic_mode", False)
        diagnostics["kalman_c_optimal"] = kalman_metadata.get("c_optimal")
        diagnostics["kalman_q_t_mean"] = kalman_metadata.get("q_t_mean")
        diagnostics["kalman_q_t_std"] = kalman_metadata.get("q_t_std")
        diagnostics["kalman_q_t_min"] = kalman_metadata.get("q_t_min")
        diagnostics["kalman_q_t_max"] = kalman_metadata.get("q_t_max")
        # Level-7+ Refinement: Robust Kalman filtering with Student-t innovations
        diagnostics["kalman_robust_t_mode"] = kalman_metadata.get("robust_t_mode", False)
        diagnostics["kalman_nu_robust"] = kalman_metadata.get("nu_robust")
        # Level-7+ Refinement: Regime-dependent drift priors
        diagnostics["kalman_regime_prior_used"] = kalman_metadata.get("regime_prior_used", False)
        diagnostics["kalman_regime_info"] = kalman_metadata.get("regime_prior_info", {})
        # phi persistence (from tuned cache or filter)
        diagnostics["kalman_phi"] = tuned_params.get("phi") if tuned_params else kalman_metadata.get("phi_used")
        diagnostics["phi_used"] = kalman_metadata.get("phi_used")

    hmm_result = feats.get("hmm_result")
    if hmm_result is not None and isinstance(hmm_result, dict):
        diagnostics["hmm_log_likelihood"] = hmm_result.get("log_likelihood", float("nan"))
        diagnostics["hmm_aic"] = hmm_result.get("aic", float("nan"))
        diagnostics["hmm_bic"] = hmm_result.get("bic", float("nan"))
        diagnostics["hmm_n_obs"] = hmm_result.get("n_obs", 0)

    nu_info = feats.get("nu_info", {})
    if isinstance(nu_info, dict):
        diagnostics["student_t_log_likelihood"] = nu_info.get("ll", float("nan"))
        diagnostics["student_t_nu"] = nu_info.get("nu_hat", float("nan"))
        diagnostics["student_t_n_obs"] = nu_info.get("n", 0)
        # Tier 2: Add standard error for posterior parameter variance tracking
        diagnostics["student_t_se_nu"] = nu_info.get("se_nu", float("nan"))

    # 2. Parameter stability tracking (expensive, only if enough data)
    ret = feats.get("ret", pd.Series(dtype=float))
    if not ret.empty and len(ret) >= 600:
        try:
            stability = track_parameter_stability(ret, window_days=252, step_days=126)
            if stability:
                diagnostics["parameter_stability"] = stability

                # Summary statistics: recent drift magnitude
                param_drift = stability.get("param_drift")
                if param_drift is not None and not param_drift.empty:
                    recent_drift = param_drift.tail(1)
                    for col in param_drift.columns:
                        val = safe_last(param_drift[col])
                        diagnostics[f"recent_{col}"] = float(val) if np.isfinite(val) else float("nan")
        except Exception:
            pass

    # 3. Out-of-sample tests (very expensive, optional)
    if enable_oos and not px.empty and len(px) >= 800:
        try:
            oos_metrics = walk_forward_validation(px, train_days=504, test_days=21, horizons=[1, 21, 63])
            if oos_metrics:
                diagnostics["out_of_sample"] = oos_metrics

                # Summary: hit rates for each horizon
                for horizon_key, metrics in oos_metrics.items():
                    if isinstance(metrics, dict):
                        hit_rate = metrics.get("hit_rate", float("nan"))
                        diagnostics[f"oos_{horizon_key}_hit_rate"] = float(hit_rate)
        except Exception:
            pass

    # 4. PIT calibration verification (Level-7: probability calibration test)
    if enable_pit_calibration and not px.empty and len(px) >= 1000:
        try:
            from calibration.pit_calibration import run_pit_calibration_test

            # Run calibration test for key horizons
            calibration_results = run_pit_calibration_test(
                px=px,
                horizons=[1, 21, 63],
                n_bins=10,
                train_days=504,
                test_days=21,
                max_predictions=500
            )

            if calibration_results:
                diagnostics["pit_calibration"] = calibration_results

                # Summary: calibration status per horizon
                for horizon, metrics in calibration_results.items():
                    diagnostics[f"pit_H{horizon}_ece"] = metrics.expected_calibration_error
                    diagnostics[f"pit_H{horizon}_calibrated"] = metrics.calibrated
                    diagnostics[f"pit_H{horizon}_diagnosis"] = metrics.calibration_diagnosis
                    diagnostics[f"pit_H{horizon}_n_predictions"] = metrics.n_predictions
        except Exception as e:
            diagnostics["pit_calibration_error"] = str(e)

    # 5. Structural model comparison (Level-7: formal falsifiability via AIC/BIC)
    if enable_model_comparison:
        try:
            from model_comparison import run_all_comparisons

            # Get required inputs
            ret = feats.get("ret", pd.Series(dtype=float))
            vol = feats.get("vol", pd.Series(dtype=float))
            garch_params = feats.get("garch_params", {})
            nu_info = feats.get("nu_info", {})
            kalman_metadata = feats.get("kalman_metadata", {})

            if not ret.empty and not vol.empty:
                # Run all model comparisons
                comparison_results = run_all_comparisons(
                    returns=ret,
                    volatility=vol,
                    garch_params=garch_params if isinstance(garch_params, dict) else None,
                    student_t_params=nu_info if isinstance(nu_info, dict) else None,
                    kalman_metadata=kalman_metadata if isinstance(kalman_metadata, dict) else None,
                )

                diagnostics["model_comparison"] = comparison_results

                # Summary: winner per category
                for category, result in comparison_results.items():
                    if result is not None and hasattr(result, 'winner_aic'):
                        diagnostics[f"model_comparison_{category}_winner_aic"] = result.winner_aic
                        diagnostics[f"model_comparison_{category}_winner_bic"] = result.winner_bic
                        diagnostics[f"model_comparison_{category}_recommendation"] = result.recommendation
        except Exception as e:
            diagnostics["model_comparison_error"] = str(e)

    return diagnostics
