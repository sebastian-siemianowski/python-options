"""
Process noise tuning: tune_asset_q, asset list management, complexity estimation.

Extracted from tune.py (Story 2.2).
"""
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from tuning.tune_modules.config import *  # noqa: F401,F403
from tuning.tune_modules.utilities import *  # noqa: F401,F403
from tuning.tune_modules.utilities import _log, _is_quiet  # noqa: E402

# Ingestion imports needed by tune_asset_q
from ingestion.data_utils import _download_prices, get_default_asset_universe
from ingestion.adaptive_quality import adaptive_data_quality


__all__ = [
    "tune_asset_q",
    "load_asset_list",
    "compute_price_data_hash",
    "get_last_price_date",
    "needs_retune",
    "stamp_tune_result",
    "estimate_tuning_complexity",
    "sort_assets_by_complexity",
    "get_optimal_worker_count",
]


# =============================================================================
# BASIC ASSET TUNING (WITHOUT FULL REGIME BMA)
# =============================================================================

def tune_asset_q(
    asset: str,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
) -> Optional[Dict]:
    """
    Tune Kalman parameters for a single asset using global model fitting.
    
    This is the basic tuning function that fits all model classes globally
    (without regime-conditional BMA). Used as:
    1. Fallback when data is insufficient for regime tuning
    2. To get global parameters for hierarchical shrinkage
    
    Args:
        asset: Asset symbol
        start_date: Start date for data
        end_date: End date (default: today)
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        
    Returns:
        Dictionary with tuned parameters and diagnostics, or None if failed
    """
    try:
        # Fetch price data (need OHLC for Garman-Klass volatility)
        # Use _download_prices ONCE to get OHLC data - extract Close from it
        # This avoids duplicate downloads (February 2026 optimization)
        df = _download_prices(asset, start_date, end_date)
        if df is None or df.empty:
            _log(f"     ⚠️  No price data for {asset}")
            return None
        
        # Adaptive data quality filter (February 2026)
        # Detects phantom/synthetic data via Volume analysis
        df, _dq_report = adaptive_data_quality(df, asset=asset, verbose=(not _is_quiet()))
        if _dq_report.get('rows_purged_leading', 0) > 0 or _dq_report.get('window_applied', False):
            _log(f"     🔬  Data quality: {_dq_report['rows_original']} → {_dq_report['rows_final']} rows")
        
        # Extract Close prices from OHLC DataFrame
        cols = {c.lower(): c for c in df.columns}
        if 'close' in cols:
            px = df[cols['close']]
        else:
            _log(f"     ⚠️  No Close column for {asset}")
            return None
        
        if px is None or len(px) < 20:
            _log(f"     ⚠️  Insufficient data for {asset}")
            return None
        
        # Compute returns
        log_ret = np.log(px / px.shift(1)).dropna()
        returns = log_ret.values
        
        # Compute volatility using Garman-Klass or HAR (7.4x more efficient than EWMA)
        vol_estimator_used = "EWMA"
        _volume_arr = None  # For Volume-based stale filter
        if GK_VOLATILITY_AVAILABLE and df is not None and not df.empty:
            try:
                # Check for OHLC columns
                cols = {c.lower(): c for c in df.columns}
                if all(c in cols for c in ['open', 'high', 'low', 'close']):
                    # Align OHLC data with returns (drop first row to match log returns)
                    df_aligned = df.iloc[1:].copy()
                    open_ = df_aligned[cols['open']].values
                    high = df_aligned[cols['high']].values
                    low = df_aligned[cols['low']].values
                    close = df_aligned[cols['close']].values
                    
                    # Extract Volume for stale-price detection (February 2026)
                    _vol_col = cols.get('volume')
                    if _vol_col is not None:
                        _volume_arr = df_aligned[_vol_col].values
                    
                    # ENFORCE HAR-GK ONLY (February 2026)
                    # HAR-GK provides multi-horizon memory for crash detection
                    # Combined with Garman-Klass (7.4x more efficient than EWMA)
                    vol, vol_estimator_used = compute_hybrid_volatility_har(
                        open_=open_, high=high, low=low, close=close,
                        span=21, annualize=False, use_har=True
                    )
                else:
                    # OHLC not available - raise error as HAR-GK is required
                    raise ValueError(f"OHLC data required for HAR-GK volatility estimation for {asset}")
            except Exception as e:
                # Log error but don't silently fall back to inferior estimator
                _log(f"     ⚠️ HAR-GK volatility estimation failed: {e}")
                raise ValueError(f"HAR-GK volatility estimation required but failed for {asset}: {e}")
        else:
            # GK/HAR module not available - this should not happen in production
            raise ImportError("HAR-GK volatility module required but not available")
        
        # Ensure returns and vol have same length
        min_len = min(len(returns), len(vol))
        returns = returns[:min_len]
        vol = vol[:min_len]

        # Remove NaN/Inf and stale-price observations (zero-return days)
        # Stale days (O=H=L=C, vol=0) produce degenerate GK variance ≈ 1e-12
        # and contaminate model parameters. Threshold 1e-10 is well below any
        # genuine trade return but catches exact zeros and float near-zeros.
        # Also filter Volume=0 phantom quotes (February 2026) — catches
        # illiquid OTC assets (GPUS) where prices move without genuine trades.
        _STALE_RETURN_THRESHOLD = 1e-10
        valid_mask = (np.isfinite(returns) & np.isfinite(vol) & (vol > 0)
                      & (np.abs(returns) > _STALE_RETURN_THRESHOLD))
        # Add Volume >= 100 filter if Volume data available
        # Skip for FX pairs (=X) and indices (^) — Yahoo reports Volume=0
        _MIN_GENUINE_VOLUME = 100  # Floor of genuine price discovery
        _skip_vol = (asset.endswith('=X') or asset.startswith('^')) if asset else False
        if _volume_arr is not None and not _skip_vol:
            _vol_aligned = _volume_arr[:min_len]
            _vol_mask = _vol_aligned >= _MIN_GENUINE_VOLUME
            n_zero_vol = int(np.sum(~_vol_mask & valid_mask))
            if n_zero_vol > 0:
                _log(f"     🧹  Filtered {n_zero_vol} additional low-volume phantom rows (Volume<{_MIN_GENUINE_VOLUME})")
            valid_mask = valid_mask & _vol_mask
        n_stale = int(np.sum(np.abs(returns) <= _STALE_RETURN_THRESHOLD))
        if n_stale > 0:
            _log(f"     🧹  Filtered {n_stale}/{len(returns)} stale-price rows ({100*n_stale/len(returns):.1f}%)")
        returns = returns[valid_mask]
        vol = vol[valid_mask]

        if len(returns) < 20:
            _log(f"     ⚠️  Insufficient valid data for {asset}")
            return None
        
        n_obs = len(returns)
        
        # Extract prices array for MR integration (February 2026)
        # Align prices with returns (skip first element since returns = diff(log(px)))
        prices_array = None
        if px is not None:
            px_values = px.values if hasattr(px, 'values') else np.array(px)
            # Skip first price to align with returns, then apply same valid_mask
            if len(px_values) > 1:
                prices_aligned = px_values[1:][:min_len]
                if len(prices_aligned) == len(valid_mask):
                    prices_array = prices_aligned[valid_mask]
                elif len(prices_aligned) >= len(returns):
                    prices_array = prices_aligned[:len(returns)]
        
        # Fit all model classes globally
        # Compute GK c prior from OHLC data (Story 2.2)
        _gk_c_prior_value = None
        if GK_C_PRIOR_AVAILABLE:
            try:
                _gk_c_prior_value = gk_c_prior(open_, high, low, close)
                if _gk_c_prior_value is not None and abs(_gk_c_prior_value - 1.0) > 0.01:
                    _log(f"     GK c prior: {_gk_c_prior_value:.3f}")
            except Exception:
                _gk_c_prior_value = None

        from tuning.tune_modules.model_fitting import fit_all_models_for_regime  # lazy: avoids circular dep
        models = fit_all_models_for_regime(
            returns, vol,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda,
            prices=prices_array,  # MR integration (February 2026)
            asset=asset,  # FIX #4: Asset-class adaptive c bounds
            gk_c_prior_value=_gk_c_prior_value,  # Story 2.2: GK-informed c prior
        )
        
        # Compute model weights using elite CRPS-dominated scoring (February 2026)
        # Only include successfully fitted models with finite metrics
        bic_values = {m: models[m].get("bic", float('inf')) for m in models if models[m].get("fit_success", False)}
        hyvarinen_values = {m: models[m].get("hyvarinen_score", float('-inf')) for m in models if models[m].get("fit_success", False)}
        crps_values = {m: models[m]["crps"] for m in models
                       if models[m].get("fit_success", False)
                       and models[m].get("crps") is not None
                       and np.isfinite(models[m]["crps"])}
        pit_pvalues = {m: models[m]["pit_ks_pvalue"] for m in models
                       if models[m].get("fit_success", False)
                       and models[m].get("pit_ks_pvalue") is not None}
        berk_pvalues = {m: models[m]["berkowitz_pvalue"] for m in models
                        if models[m].get("fit_success", False)
                        and models[m].get("berkowitz_pvalue") is not None}
        mad_values = {m: models[m]["histogram_mad"] for m in models
                      if models[m].get("fit_success", False)
                      and models[m].get("histogram_mad") is not None}
        berk_lr_values = {m: models[m]["berkowitz_lr"] for m in models
                          if models[m].get("fit_success", False)
                          and models[m].get("berkowitz_lr") is not None}
        pit_count_values = {m: models[m]["pit_count"] for m in models
                            if models[m].get("fit_success", False)
                            and models[m].get("pit_count") is not None}
        ad_pvalues_global = {m: models[m]["ad_pvalue"] for m in models
                             if models[m].get("fit_success", False)
                             and models[m].get("ad_pvalue") is not None
                             and np.isfinite(models[m]["ad_pvalue"])}
        
        # Use elite CRPS-dominated scoring with PIT/Berk/MAD penalties
        if crps_values and CRPS_SCORING_ENABLED:
            model_weights, weight_meta = compute_regime_aware_model_weights(
                bic_values, hyvarinen_values, crps_values,
                pit_pvalues=pit_pvalues, berk_pvalues=berk_pvalues,
                berkowitz_lr_stats=berk_lr_values, pit_counts=pit_count_values,
                mad_values=mad_values, ad_pvalues=ad_pvalues_global, regime=None
            )
        else:
            model_weights = compute_bic_model_weights(bic_values)
            weight_meta = {"scoring_method": "bic_only", "crps_enabled": False, "pit_enabled": False}
        
        # Store standardized scores and weights in each model
        for m in models:
            w = model_weights.get(m, 1e-10)
            models[m]['model_weight_entropy'] = float(w)
            if weight_meta:
                combined_score_val = weight_meta.get('combined_scores_standardized', {}).get(m)
                models[m]['combined_score'] = float(combined_score_val) if combined_score_val is not None else 0.0
                bic_std_val = weight_meta.get('bic_standardized', {}).get(m)
                models[m]['standardized_bic'] = float(bic_std_val) if bic_std_val is not None else None
                hyv_std_val = weight_meta.get('hyvarinen_standardized', {}).get(m)
                models[m]['standardized_hyvarinen'] = float(hyv_std_val) if hyv_std_val is not None else None
                crps_std_val = weight_meta.get('crps_standardized', {}).get(m)
                models[m]['standardized_crps'] = float(crps_std_val) if crps_std_val is not None else None
                scoring_weights = weight_meta.get('weights_used', {})
                models[m]['scoring_weights'] = {
                    'bic': float(scoring_weights.get('bic', 0.0)),
                    'hyvarinen': float(scoring_weights.get('hyvarinen', 0.0)),
                    'crps': float(scoring_weights.get('crps', 0.0)),
                }
                models[m]['crps_scoring_enabled'] = weight_meta.get('crps_enabled', False)
        
        # Find best model by WEIGHT (after calibration veto gate)
        # The veto gate forces catastrophically miscalibrated models (PIT<0.01 or
        # Berk<0.01) to floor weight, redistributing to well-calibrated models.
        # Selecting by weight ensures the winner is always well-calibrated.
        best_model = max(
            ((m, w) for m, w in model_weights.items() if w is not None),
            key=lambda x: x[1]
        )[0]
        best_params = models[best_model]
        
        # Gate external augmentation layers for unified models (they already have internal calibration)
        is_unified_winner = "unified" in best_model.lower() if best_model else False

        # GMM: Removed (empirically — bimodality hypothesis consistently rejected)
        # Hansen Skew-T: Now handled internally by Stage 7.5 / Stage U-H
        hansen_skew_t_result = None
        hansen_skew_t_diagnostics = None
        hansen_comparison = None
        
        # =====================================================================
        # FIT EVT/GPD DISTRIBUTION (Extreme Value Theory for Tail Risk)
        # =====================================================================
        # The Pickands–Balkema–de Haan theorem provides theoretical foundation:
        # exceedances over high threshold u converge to GPD distribution.
        #
        # This pre-computes optimal EVT parameters per asset for use in signals.py
        # where EVT-corrected expected loss is used for position sizing.
        #
        # Key outputs:
        #   - ξ (xi): GPD shape parameter (ξ > 0 = heavy tails, ξ = 1/ν for Student-t)
        #   - σ (sigma): GPD scale parameter
        #   - u (threshold): POT threshold (90th percentile default)
        #   - CTE: Conditional Tail Expectation = E[Loss | Loss > u]
        # =====================================================================
        evt_result = None
        evt_diagnostics = None
        evt_consistency = None
        
        if EVT_AVAILABLE and not is_unified_winner:
            try:
                # Compute losses (positive values)
                losses = -returns[returns < 0]
                
                if len(losses) >= EVT_MIN_EXCEEDANCES:
                    gpd_result = fit_gpd_pot(
                        losses,
                        threshold_percentile=EVT_THRESHOLD_PERCENTILE_DEFAULT,
                        method='auto'
                    )
                    
                    if gpd_result.fit_success:
                        evt_result = gpd_result.to_dict()
                        evt_diagnostics = {
                            "fit_success": True,
                            "n_losses": len(losses),
                            "n_total_obs": n_obs,
                        }
                        
                        # Check consistency with Student-t ν
                        best_nu = best_params.get("nu")
                        if best_nu is not None:
                            evt_consistency = check_student_t_consistency(best_nu, gpd_result.xi)
                            evt_diagnostics["student_t_consistency"] = evt_consistency
                        
                        # Log result
                        xi = gpd_result.xi
                        implied_nu = gpd_result.implied_student_t_nu
                        tail_type = "heavy" if xi > 0.2 else ("moderate" if xi > 0.05 else "light")
                        nu_str = f"(≈ν={implied_nu:.0f})" if implied_nu and implied_nu < 100 else ""
                        _log(f"     ✓ EVT/GPD: ξ={xi:.3f} {nu_str} [{tail_type} tails], "
                             f"CTE={gpd_result.cte:.4f}, n_exc={gpd_result.n_exceedances}")
                    else:
                        error_msg = gpd_result.diagnostics.get("error", "unknown")
                        _log(f"     ⚠️ EVT/GPD fit failed: {error_msg}")
                        evt_diagnostics = {
                            "fit_success": False,
                            "error": error_msg,
                            "n_losses": len(losses),
                        }
                else:
                    _log(f"     ⚠️ EVT/GPD skipped: insufficient losses ({len(losses)} < {EVT_MIN_EXCEEDANCES})")
                    evt_diagnostics = {
                        "fit_success": False,
                        "error": "insufficient_losses",
                        "n_losses": len(losses),
                    }
            except Exception as evt_err:
                _log(f"     ⚠️ EVT/GPD fitting exception: {evt_err}")
                evt_diagnostics = {"fit_success": False, "error": str(evt_err)}
        else:
            evt_diagnostics = {"fit_success": False, "error": "evt_not_available"}
        
        # =====================================================================
        # CONTAMINATED STUDENT-T: Now handled internally by Stage 7.6 / Stage U-C
        # Global fit block removed (was dead code under UNIFIED_STUDENT_T_ONLY)
        # =====================================================================
        cst_result = None
        cst_diagnostics = None
        cst_comparison = None
        
        # =====================================================================
        # PIT-DRIVEN ESCALATION: ν-REFINEMENT (L1 → L2)
        # =====================================================================
        nu_refinement_result = None
        nu_refinement_attempted = False
        nu_refinement_improved = False
        
        if ADAPTIVE_NU_AVAILABLE and ADAPTIVE_NU_ENABLED:
            pit_pvalue = best_params.get("pit_ks_pvalue", 1.0)
            best_nu = best_params.get("nu")
            is_student_t = best_model.startswith("phi_student_t") if best_model else False
            pit_fails = pit_pvalue < ADAPTIVE_NU_PIT_THRESHOLD
            pit_severe = pit_pvalue < ADAPTIVE_NU_PIT_SEVERE_THRESHOLD
            
            if is_student_t and best_nu and (pit_fails or pit_severe):
                nu_refinement_attempted = True
                try:
                    candidates = ADAPTIVE_NU_CANDIDATES.get(float(best_nu), [])
                    if candidates:
                        _log(f"     🔄 ν-refinement: PIT p={pit_pvalue:.4f} → testing ν={candidates}")
                        best_refined_nu = best_nu
                        best_refined_pit = pit_pvalue
                        
                        for nu_candidate in candidates:
                            model_key = f"phi_student_t_nu_{int(nu_candidate)}"
                            if model_key in models and models[model_key].get("fit_success"):
                                cand_pit = models[model_key].get("pit_ks_pvalue", 0)
                                if cand_pit > best_refined_pit:
                                    best_refined_nu = nu_candidate
                                    best_refined_pit = cand_pit
                                    best_params = models[model_key]
                                    best_model = model_key
                                    _log(f"        ✓ ν={nu_candidate}: PIT p={cand_pit:.4f} (improved)")
                        
                        if best_refined_pit > pit_pvalue:
                            nu_refinement_improved = True
                            _log(f"     ✓ ν-refinement SUCCESS: PIT {pit_pvalue:.4f}→{best_refined_pit:.4f}")
                        
                        nu_refinement_result = {
                            "refinement_attempted": True,
                            "nu_original": best_nu,
                            "nu_final": best_refined_nu,
                            "improvement_achieved": nu_refinement_improved,
                            "pit_before": pit_pvalue,
                            "pit_after": best_refined_pit,
                        }
                except Exception as nu_err:
                    _log(f"     ⚠️ ν-refinement error: {nu_err}")
                    nu_refinement_result = {"error": str(nu_err)}
        
        # =====================================================================
        # PIT-DRIVEN ESCALATION: EVT TAIL SPLICE (L3)
        # =====================================================================
        # EVT Tail Splice replaces the tail portion of the CDF with GPD.
        # This is triggered when:
        #   1. PIT calibration still fails after ν-refinement
        #   2. EVT has been fitted successfully
        #   3. EVT splice improves PIT p-value by >= 50%
        #
        # The spliced distribution provides theoretically justified tail
        # extrapolation via the Pickands–Balkema–de Haan theorem.
        # =====================================================================
        evt_splice_result = None
        evt_splice_attempted = False
        evt_splice_selected = False
        
        if EVT_SPLICE_AVAILABLE and EVT_SPLICE_ENABLED and evt_result is not None:
            # Check if PIT still fails after previous escalations
            current_pit = best_params.get("pit_ks_pvalue", 1.0)
            pit_still_fails = current_pit < 0.05
            
            if pit_still_fails:
                evt_splice_attempted = True
                try:
                    # Reconstruct GPDFitResult from stored dict
                    gpd_result_obj = GPDFitResult.from_dict(evt_result)
                    
                    # Get filter outputs for PIT computation
                    # We need to run filter with best params
                    q_best = best_params.get("q", 1e-6)
                    c_best = best_params.get("c", 1.0)
                    phi_best = best_params.get("phi", 0.0)
                    nu_best = best_params.get("nu")
                    
                    # Run filter
                    if phi_best is not None:
                        from models import PhiStudentTDriftModel, PhiGaussianDriftModel
                        if nu_best is not None:
                            mu_filt, P_filt, _ = PhiStudentTDriftModel.filter_phi(
                                returns, vol, q_best, c_best, phi_best, nu_best
                            )
                        else:
                            mu_filt, P_filt, _ = PhiGaussianDriftModel.filter_phi(
                                returns, vol, q_best, c_best, phi_best
                            )
                    else:
                        from models import GaussianDriftModel
                        mu_filt, P_filt, _ = GaussianDriftModel.filter(
                            returns, vol, q_best, c_best
                        )
                    
                    # Test if EVT splice improves PIT
                    should_select, evt_pit_pvalue, improvement_ratio, evt_diag = test_evt_splice_improvement(
                        returns=returns,
                        mu_filtered=mu_filt,
                        vol=vol,
                        P_filtered=P_filt,
                        c=c_best,
                        nu=nu_best,
                        gpd_result=gpd_result_obj,
                        baseline_pit_pvalue=current_pit,
                    )
                    
                    evt_splice_result = {
                        "attempted": True,
                        "should_select": should_select,
                        "baseline_pit_pvalue": current_pit,
                        "evt_pit_pvalue": evt_pit_pvalue,
                        "improvement_ratio": improvement_ratio,
                        "diagnostics": evt_diag,
                    }
                    
                    if should_select:
                        evt_splice_selected = True
                        # Update best params to flag EVT splice
                        best_params = best_params.copy()
                        best_params["evt_splice_selected"] = True
                        best_params["evt_splice_pit_pvalue"] = evt_pit_pvalue
                        best_params["pit_ks_pvalue"] = evt_pit_pvalue  # Update PIT
                        _log(f"     ✓ EVT Splice L3: PIT {current_pit:.4f}→{evt_pit_pvalue:.4f} ({improvement_ratio:.1f}x improvement)")
                    else:
                        reason = evt_diag.get("reason", "unknown")
                        _log(f"     ⚠️ EVT Splice L3: Not selected ({reason})")
                        
                except Exception as evt_err:
                    _log(f"     ⚠️ EVT Splice error: {evt_err}")
                    evt_splice_result = {"attempted": True, "error": str(evt_err)}

        
        # Build result structure - BMA-compatible format
        # signals.py expects: {"global": {...}, "has_bma": True}
        global_data = {
            "asset": asset,
            "q": float(best_params.get("q", 1e-6)),
            "c": float(best_params.get("c", 1.0)),
            "phi": best_params.get("phi"),
            "nu": best_params.get("nu"),
            "noise_model": best_model,
            "best_model": best_model,  # Selected by max weight (after calibration veto gate)
            # RV-Q specific parameters (proactive vol-adaptive noise)
            "q_base": best_params.get("q_base"),
            "gamma": best_params.get("gamma"),
            "rv_q_model": best_params.get("rv_q_model", False),
            # Unified Student-t specific parameters (February 2026 - Elite Architecture)
            "unified_model": best_params.get("unified_model", False),
            "gaussian_unified": best_params.get("gaussian_unified", False),
            "alpha_asym": best_params.get("alpha_asym"),
            "gamma_vov": best_params.get("gamma_vov"),
            "ms_sensitivity": best_params.get("ms_sensitivity"),
            "q_stress_ratio": best_params.get("q_stress_ratio"),
            "vov_damping": best_params.get("vov_damping"),
            "degraded": best_params.get("degraded", False),
            "hessian_cond": best_params.get("hessian_cond"),
            "pit_calibration_grade": best_params.get("pit_calibration_grade"),
            "bic": float(best_params.get("bic", float('inf'))),
            "aic": float(best_params.get("aic", float('inf'))),
            "log_likelihood": float(best_params.get("log_likelihood", float('-inf'))),
            "mean_log_likelihood": float(best_params.get("mean_log_likelihood", float('-inf'))),
            "ks_statistic": float(best_params.get("ks_statistic", 0.0)),
            "pit_ks_pvalue": float(best_params.get("pit_ks_pvalue", 0.0)),
            "calibration_warning": best_params.get("pit_ks_pvalue", 1.0) < 0.05,
            "n_obs": n_obs,
            # Volatility estimator used (February 2026 - Garman-Klass support)
            "volatility_estimator": vol_estimator_used,
            "model_weights": model_weights,
            "model_posterior": model_weights,  # BMA expects model_posterior
            "models": models,  # Full model details for BMA
            "model_comparison": {m: {
                "ll": models[m].get("log_likelihood", float('-inf')),
                "bic": models[m].get("bic", float('inf')),
                "aic": models[m].get("aic", float('inf')),
                "fit_success": models[m].get("fit_success", False),
            } for m in models},
            # GMM parameters — removed (bimodality hypothesis rejected)
            # Hansen Skew-t parameters for asymmetric tail modeling
            "hansen_skew_t": hansen_skew_t_result,
            "hansen_skew_t_diagnostics": hansen_skew_t_diagnostics,
            "hansen_vs_symmetric_comparison": hansen_comparison,
            # EVT/GPD parameters for tail risk modeling
            "evt": evt_result,
            "evt_diagnostics": evt_diagnostics,
            "evt_student_t_consistency": evt_consistency,
            # EVT Tail Splice for PIT calibration (L3 escalation)
            "evt_splice": evt_splice_result,
            "evt_splice_attempted": evt_splice_attempted,
            "evt_splice_selected": evt_splice_selected,
            # Contaminated Student-t Mixture for regime-dependent tails
            "contaminated_student_t": cst_result,
            "contaminated_student_t_diagnostics": cst_diagnostics,
            "contaminated_vs_single_comparison": cst_comparison,
            # PIT-driven ν-refinement results (L2 escalation)
            "nu_refinement": nu_refinement_result,
            "nu_refinement_attempted": nu_refinement_attempted,
            "nu_refinement_improved": nu_refinement_improved,
            # Market conditioning flags (February 2026)
            # These tell signals.py to apply VIX-based ν adjustment at inference time
            "market_conditioning_enabled": MARKET_CONDITIONING_ENABLED and MARKET_CONDITIONING_AVAILABLE,
            "vix_nu_adjustment_enabled": MARKET_CONDITIONING_ENABLED and MARKET_CONDITIONING_AVAILABLE and best_params.get("nu") is not None,
        }
        
        result = {
            "asset": asset,
            "has_bma": True,  # CRITICAL: signals.py checks this flag to accept the cache
            "global": global_data,  # BMA-compatible structure
            "regime": None,  # No regime data for basic tuning
            "use_regime_tuning": False,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        
        return result
        
    except Exception as e:
        import traceback
        _log(f"     ❌ {asset}: Failed - {e}")
        traceback.print_exc()
        return None


# =============================================================================
# ENTROPY REGULARIZATION CONSTANTS — IMPORTED FROM tuning.diagnostics
# =============================================================================
# DEFAULT_ENTROPY_LAMBDA and DEFAULT_MIN_WEIGHT_FRACTION are now imported
# from tuning.diagnostics to avoid duplication.
# =============================================================================


def load_asset_list(assets_arg: Optional[str], assets_file: Optional[str]) -> List[str]:
    """Load list of assets from command-line argument or file."""
    if assets_arg:
        return [a.strip() for a in assets_arg.split(',') if a.strip()]
    
    if assets_file and os.path.exists(assets_file):
        with open(assets_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Default asset list: use centralized universe from fx_data_utils
    return get_default_asset_universe()


# =============================================================================
# Story 3.1: Content-Based Change Detection for Incremental Tuning
# =============================================================================
# Uses hash of last 20 rows of price CSV to detect new data.
# NOT based on mtime (unreliable across git checkout, rsync, backup restore).
# =============================================================================

def compute_price_data_hash(symbol: str, prices_dir: str = None) -> Optional[str]:
    """
    Story 3.1: Compute content-based hash of last 20 rows of price CSV.
    
    Uses hashlib.sha256 on the raw bytes of the last 20 lines.
    Returns hex digest or None if file not found.
    """
    import hashlib
    
    if prices_dir is None:
        prices_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data", "prices")
    
    price_file = os.path.join(prices_dir, f"{symbol}_1d.csv")
    if not os.path.exists(price_file):
        return None
    
    try:
        with open(price_file, 'rb') as f:
            lines = f.readlines()
        # Hash last 20 lines (or all if fewer)
        tail = lines[-20:] if len(lines) > 20 else lines
        return hashlib.sha256(b''.join(tail)).hexdigest()
    except Exception:
        return None


def get_last_price_date(symbol: str, prices_dir: str = None) -> Optional[str]:
    """
    Story 3.1: Get the date of the last price row.
    """
    if prices_dir is None:
        prices_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data", "prices")
    
    price_file = os.path.join(prices_dir, f"{symbol}_1d.csv")
    if not os.path.exists(price_file):
        return None
    
    try:
        import csv
        with open(price_file, 'r') as f:
            lines = f.readlines()
        if len(lines) < 2:
            return None
        # Last line, first column (date)
        last_line = lines[-1].strip()
        if last_line:
            return last_line.split(',')[0]
    except Exception:
        pass
    return None


def needs_retune(symbol: str, cached_params: Optional[dict], prices_dir: str = None) -> bool:
    """
    Story 3.1: Check if an asset needs re-tuning based on content hash.
    
    Returns True if:
      - No cached params exist
      - Cached params don't have a price_data_hash
      - Price data hash has changed since last tune
    """
    if cached_params is None:
        return True
    
    stored_hash = None
    if isinstance(cached_params, dict):
        global_block = cached_params.get("global", cached_params)
        stored_hash = global_block.get("price_data_hash")
    
    if stored_hash is None:
        return True
    
    current_hash = compute_price_data_hash(symbol, prices_dir)
    if current_hash is None:
        return False  # No price file -> can't tune
    
    return current_hash != stored_hash


def stamp_tune_result(result: dict, symbol: str, prices_dir: str = None) -> dict:
    """
    Story 3.1: Add price_data_hash and last_price_date to tune result.
    """
    if result is None:
        return result
    
    current_hash = compute_price_data_hash(symbol, prices_dir)
    last_date = get_last_price_date(symbol, prices_dir)
    
    global_block = result.get("global", result)
    if current_hash:
        global_block["price_data_hash"] = current_hash
    if last_date:
        global_block["last_price_date"] = last_date
    
    return result


# =============================================================================
# Story 3.2: Parallel Tuning Optimization
# =============================================================================

def estimate_tuning_complexity(symbol: str, prices_dir: str = None) -> float:
    """
    Story 3.2: Estimate tuning complexity for work-stealing optimization.
    
    Higher score = more complex (should be tuned first for load balancing).
    Based on: data length, annualized vol (volatile assets have more models to try).
    
    Returns estimated complexity score (0-100).
    """
    if prices_dir is None:
        prices_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data", "prices")
    
    price_file = os.path.join(prices_dir, f"{symbol}_1d.csv")
    if not os.path.exists(price_file):
        return 50.0  # Default complexity
    
    try:
        with open(price_file, 'r') as f:
            line_count = sum(1 for _ in f)
        # Length score: more data = more work
        length_score = min(line_count / 30, 50)
        
        # Vol score: approximate from last 20 lines
        with open(price_file, 'r') as f:
            lines = f.readlines()
        if len(lines) < 22:
            return length_score
        
        closes = []
        for line in lines[-21:]:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                try:
                    closes.append(float(parts[4]))
                except (ValueError, IndexError):
                    continue
        
        if len(closes) >= 10:
            import numpy as np
            rets = np.diff(np.log(np.array(closes)))
            vol = float(np.std(rets) * np.sqrt(252))
            vol_score = min(vol / 0.01, 50)  # High vol = more complexity
            return length_score + vol_score
        
        return length_score
    except Exception:
        return 50.0


def sort_assets_by_complexity(assets: list, prices_dir: str = None) -> list:
    """
    Story 3.2: Sort assets by estimated complexity, slowest first.
    
    This enables work-stealing: slow assets start first, fast ones fill gaps.
    """
    scored = [(a, estimate_tuning_complexity(a, prices_dir)) for a in assets]
    scored.sort(key=lambda x: -x[1])  # Descending: most complex first
    return [a for a, _ in scored]


def get_optimal_worker_count() -> int:
    """Story 3.2: Optimal worker count for parallel tuning."""
    cpu = os.cpu_count() or 4
    return min(cpu - 1, 8)  # Leave 1 core for OS, cap at 8


