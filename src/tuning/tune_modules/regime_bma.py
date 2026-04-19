"""
Regime BMA: fit_regime_model_posterior, tune_regime_model_averaging.

Extracted from tune.py (Story 4.2). Contains the Bayesian Model Averaging
engine that computes per-regime model posteriors and the orchestrator that
coordinates global + per-regime model fitting with hierarchical shrinkage.
"""
import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.stats import norm, t as student_t

from tuning.tune_modules.config import *  # noqa: F401,F403
from tuning.tune_modules.utilities import *  # noqa: F401,F403
from tuning.tune_modules.utilities import _log  # noqa: E402
from tuning.tune_modules.model_fitting import fit_all_models_for_regime  # noqa: E402
from tuning.tune_modules.calibration_pipeline import apply_regime_q_floor  # noqa: E402


__all__ = [
    "fit_regime_model_posterior",
    "tune_regime_model_averaging",
]


def fit_regime_model_posterior(
    returns: np.ndarray,
    vol: np.ndarray,
    regime_labels: np.ndarray,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
    min_samples: int = MIN_REGIME_SAMPLES,
    temporal_alpha: float = DEFAULT_TEMPORAL_ALPHA,
    previous_posteriors: Optional[Dict[int, Dict[str, float]]] = None,
    global_models: Optional[Dict[str, Dict]] = None,
    global_posterior: Optional[Dict[str, float]] = None,
    model_selection_method: str = DEFAULT_MODEL_SELECTION_METHOD,
    bic_weight: float = DEFAULT_BIC_WEIGHT,
    prices: np.ndarray = None,  # Added for MR integration (February 2026)
    asset: str = None,  # FIX #4: Asset symbol for c-bounds detection
    gk_c_prior_value: float = None,  # Story 2.2: GK-informed c prior
) -> Dict[int, Dict]:
    """
    Compute regime-conditional Bayesian model averaging with temporal smoothing.
    
    This function implements the core epistemic law:
    
        p(r_{t+H} | r) = Σ_m p(r_{t+H} | r, m, θ_{r,m}) · p(m | r)
    
    For EACH regime r:
    1. Fit EACH candidate model class m independently
    2. Compute mean_log_likelihood, BIC, AIC, Hyvärinen score for each (r, m)
    3. Convert scores to posterior weights using specified method:
       - 'bic': w_raw(m|r) = exp(-0.5 * ΔBIC)
       - 'hyvarinen': w_raw(m|r) = exp(ΔH) where H is negated Hyvärinen score
       - 'combined': geometric mean of BIC and Hyvärinen weights
    4. Apply temporal smoothing: w_smooth = prev_p^alpha * w_raw
    5. Normalize to get p(m|r)
    
    HIERARCHICAL FALLBACK:
    When a regime r has insufficient samples:
    - Use global_models as the regime's models (hierarchical borrowing)
    - Use global_posterior as the regime's model_posterior
    - Mark as fallback with borrowed_from_global=True
    
    This is correct hierarchical Bayesian shrinkage:
        p(m|r) = p(m|global) when data is insufficient
        θ_{r,m} = θ_{global,m} when data is insufficient
    
    CRITICAL RULES:
    - Never select a single best model per regime
    - Never discard models
    - Never force weights to zero
    - Never return empty models for a regime
    - Never mix tuning with signal logic
    - Preserve all priors, shrinkage, diagnostics
    
    Args:
        returns: Array of returns
        vol: Array of EWMA volatility  
        regime_labels: Array of regime labels (0-4) for each time step
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        min_samples: Minimum samples required per regime
        temporal_alpha: Smoothing exponent for model posterior evolution
        previous_posteriors: Previous model posteriors per regime (for smoothing)
        global_models: Global model fits (for hierarchical fallback)
        global_posterior: Global model posterior (for hierarchical fallback)
        model_selection_method: Method for computing model weights:
            - 'bic': Use BIC only (traditional)
            - 'hyvarinen': Use Hyvärinen score only (robust to misspecification)
            - 'combined': Geometric mean of BIC and Hyvärinen weights (default)
        bic_weight: Weight for BIC when using 'combined' method (0-1)
        
    Returns:
        Dictionary with regime-conditional model posteriors and parameters:
        {
            r: {
                "model_posterior": { m: p(m|r) },
                "models": {
                    m: {
                        "q", "phi", "nu", "c",
                        "mean_log_likelihood",
                        "bic", "aic",
                        "ks_statistic", "pit_ks_pvalue",
                        "fit_success", ...
                    }
                },
                "regime_meta": {
                    "temporal_alpha": α,
                    "n_samples": N,
                    "regime_name": str,
                    "fallback": bool,
                    "borrowed_from_global": bool,
                    "shrinkage_applied": bool
                }
            }
        }
    """
    # Validate inputs
    returns = np.asarray(returns).flatten()
    vol = np.asarray(vol).flatten()
    regime_labels = np.asarray(regime_labels).flatten().astype(int)
    
    if len(returns) != len(regime_labels):
        raise ValueError(f"Length mismatch: returns={len(returns)}, regime_labels={len(regime_labels)}")
    
    # Initialize result structure
    regime_results = {}
    
    # Process each regime
    for regime in range(5):
        regime_name = REGIME_LABELS.get(regime, f"REGIME_{regime}")
        mask = (regime_labels == regime)
        n_samples = int(np.sum(mask))
        
        _log(f"  📊 Fitting all models for {regime_name} (n={n_samples})...")
        
        # Get previous posterior for this regime (for temporal smoothing)
        prev_posterior = None
        if previous_posteriors is not None and regime in previous_posteriors:
            prev_posterior = previous_posteriors[regime]
        
        # Check if we have enough samples
        if n_samples < min_samples:
            _log(f"     ⚠️  Insufficient samples ({n_samples} < {min_samples}), using hierarchical fallback from global")
            # =========================================================================
            # HIERARCHICAL BAYESIAN FALLBACK
            # =========================================================================
            # When regime r has insufficient samples, we borrow from global:
            #   p(m|r) = p(m|global)
            #   θ_{r,m} = θ_{global,m}
            #
            # This is correct hierarchical Bayesian shrinkage, not parameter invention.
            # Never return empty models - always provide usable fallback.
            # =========================================================================
            if global_models is not None and global_posterior is not None:
                # Use global as hierarchical fallback
                regime_results[regime] = {
                    "model_posterior": global_posterior.copy(),
                    "models": global_models.copy(),
                    "regime_meta": {
                        "temporal_alpha": temporal_alpha,
                        "n_samples": n_samples,
                        "regime_name": regime_name,
                        "fallback": True,
                        "borrowed_from_global": True,
                        "fallback_reason": f"insufficient_samples_{n_samples}_lt_{min_samples}",
                    }
                }
            else:
                # =====================================================================
                # CRITICAL: No global available - this should NOT happen in normal flow
                # =====================================================================
                # tune_regime_model_averaging() always computes global first.
                # If we reach here, it's a programming error or corrupt state.
                #
                # DO NOT synthesize fake models. That violates Bayesian integrity.
                # Instead: skip this regime and let it be handled upstream.
                #
                # The correct response to missing evidence is ignorance, not invention.
                # =====================================================================
                _log(f"     ⚠️  CRITICAL: No global models for regime {regime} fallback - skipping")
                # Skip this regime - it will be missing from regime_results
                # Downstream must handle missing regimes by using global directly
            continue
        
        # Extract regime-specific data
        ret_regime = returns[mask]
        vol_regime = vol[mask]
        
        # Extract regime-specific prices for MR integration (February 2026)
        prices_regime = prices[mask] if prices is not None else None
        regime_labels_regime = regime_labels[mask] if regime_labels is not None else None
        
        # =====================================================================
        # Step 1: Fit ALL models for this regime
        # =====================================================================
        models = fit_all_models_for_regime(
            ret_regime, vol_regime,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda,
            prices=prices_regime,  # MR integration (February 2026)
            regime_labels=regime_labels_regime,
            asset=asset,  # FIX #4: Asset-class adaptive c bounds
            gk_c_prior_value=gk_c_prior_value,  # Story 2.2
        )

        # =====================================================================
        # Step 1b: Apply regime-conditional q floor (Story 1.1, April 2026)
        # Story 1.7: Vol-proportional q floor
        # =====================================================================
        _asset_vol_ann = float(np.median(vol_regime)) * np.sqrt(252) if len(vol_regime) > 0 else 0.0
        n_floored, n_total = apply_regime_q_floor(
            models, regime, ret_regime, vol_regime,
            asset_vol=_asset_vol_ann,
        )
        _q_floor_val = compute_vol_proportional_q_floor(regime, _asset_vol_ann) if _asset_vol_ann > 0 else Q_FLOOR_BY_REGIME.get(regime, 0)
        if n_floored > 0:
            _log(f"     Q-floor applied: {n_floored}/{n_total} models "
                 f"(floor={_q_floor_val:.1e} for {regime_name}, vol={_asset_vol_ann:.2%})")

        # =====================================================================
        # Step 2: Extract BIC, Hyvärinen, CRPS, PIT and compute LFO-CV scores
        # =====================================================================
        bic_values = {m: models[m].get("bic", float('inf')) for m in models}
        hyvarinen_scores = {m: models[m].get("hyvarinen_score", float('-inf')) for m in models}
        crps_values = {m: models[m].get("crps", float('inf')) for m in models if models[m].get("crps") is not None}
        # February 2026 - Elite PIT calibration: extract PIT p-values for regime-aware scoring
        pit_pvalues = {m: models[m].get("pit_ks_pvalue") for m in models if models[m].get("pit_ks_pvalue") is not None}
        berk_pvalues = {m: models[m]["berkowitz_pvalue"] for m in models
                        if models[m].get("fit_success", False)
                        and models[m].get("berkowitz_pvalue") is not None}
        mad_values = {m: models[m]["histogram_mad"] for m in models
                      if models[m].get("fit_success", False)
                      and models[m].get("histogram_mad") is not None}
        berk_lr_regime = {m: models[m]["berkowitz_lr"] for m in models
                         if models[m].get("fit_success", False)
                         and models[m].get("berkowitz_lr") is not None}
        pit_count_regime = {m: models[m]["pit_count"] for m in models
                           if models[m].get("fit_success", False)
                           and models[m].get("pit_count") is not None}
        ad_pvalues_regime = {m: models[m]["ad_pvalue"] for m in models
                            if models[m].get("fit_success", False)
                            and models[m].get("ad_pvalue") is not None
                            and np.isfinite(models[m]["ad_pvalue"])}
        
        # LFO-CV scores for proper out-of-sample model selection (February 2026)
        lfo_cv_scores = {}
        if LFO_CV_ENABLED and n_samples >= 50:  # Need enough data for LFO-CV
            import time as _lfo_time_mod
            _lfo_t0 = _lfo_time_mod.perf_counter()
            _lfo_computed = 0
            _lfo_cached = 0
            for m, info in models.items():
                if info.get("fit_success", False):
                    # Check if model already has LFO-CV score (e.g., MS-q models)
                    if info.get("lfo_cv_score") is not None:
                        lfo_cv_scores[m] = info["lfo_cv_score"]
                        _lfo_cached += 1
                    else:
                        # Compute LFO-CV score
                        q_val = info.get("q", 1e-6)
                        c_val = info.get("c", 1.0)
                        phi_val = info.get("phi", 1.0) if info.get("phi") is not None else 1.0
                        nu_val = info.get("nu")
                        
                        try:
                            if nu_val is not None:
                                lfo_score, lfo_diag = compute_lfo_cv_score_student_t(
                                    ret_regime, vol_regime, q_val, c_val, phi_val, nu_val,
                                    min_train_frac=LFO_CV_MIN_TRAIN_FRAC
                                )
                            else:
                                lfo_score, lfo_diag = compute_lfo_cv_score_gaussian(
                                    ret_regime, vol_regime, q_val, c_val, phi_val,
                                    min_train_frac=LFO_CV_MIN_TRAIN_FRAC
                                )
                            
                            lfo_cv_scores[m] = lfo_score
                            models[m]["lfo_cv_score"] = float(lfo_score)
                            models[m]["lfo_cv_diagnostics"] = lfo_diag
                            _lfo_computed += 1
                        except Exception as e:
                            lfo_cv_scores[m] = float('-inf')
                            models[m]["lfo_cv_error"] = str(e)
            _lfo_elapsed = _lfo_time_mod.perf_counter() - _lfo_t0
            _log(f"     LFO-CV: {_lfo_computed} computed, {_lfo_cached} cached in {_lfo_elapsed:.2f}s")
        
        # Print model fits
        for m, info in models.items():
            if info.get("fit_success", False):
                bic_val = info.get("bic", float('nan'))
                hyv_val = info.get("hyvarinen_score", float('nan'))
                crps_val = info.get("crps", float('nan'))
                mean_ll = info.get("mean_log_likelihood", float('nan'))
                lfo_val = info.get("lfo_cv_score", float('nan'))
                if LFO_CV_ENABLED and np.isfinite(lfo_val):
                    _log(f"     {m}: BIC={bic_val:.1f}, H={hyv_val:.4f}, CRPS={crps_val:.4f}, LFO={lfo_val:.4f}")
                else:
                    _log(f"     {m}: BIC={bic_val:.1f}, H={hyv_val:.4f}, CRPS={crps_val:.4f}, mean_LL={mean_ll:.4f}")
            else:
                _log(f"     {m}: FAILED - {info.get('error', 'unknown')}")
        
        # =====================================================================
        # Step 3: Compute raw weights using regime-aware method (February 2026)
        # =====================================================================
        # Model selection uses elite CRPS-dominated 6-component scoring:
        #   Score = w_crps × CRPS_std + w_pit × PIT_dev_std + w_berk × Berk_std
        #         + w_tail × Tail_std + w_mad × MAD_std + w_ad × AD_dev_std
        #
        # All components are robustly standardized via median/MAD (winsorized ±5σ).
        # BIC and Hyvärinen are stored as metadata but NOT part of selection score.
        # Regime-specific weight profiles adjust component emphasis (see diagnostics.py).
        #
        # SMALL SAMPLE HANDLING: When Hyvärinen is disabled, use BIC+CRPS only
        # =====================================================================
        hyvarinen_disabled = n_samples < MIN_HYVARINEN_SAMPLES
        weight_metadata = None
        
        if hyvarinen_disabled:
            # Small samples: use BIC + CRPS + PIT only (Hyvärinen unreliable)
            raw_weights, weight_metadata = compute_regime_aware_model_weights(
                bic_values, hyvarinen_scores, crps_values,
                pit_pvalues=pit_pvalues,  # February 2026 - Elite PIT calibration
                berk_pvalues=berk_pvalues,
                ad_pvalues=ad_pvalues_regime,  # AD veto gate
                berkowitz_lr_stats=berk_lr_regime, pit_counts=pit_count_regime,
                mad_values=mad_values,
                regime=regime, 
                bic_weight=0.50, hyvarinen_weight=0.0, crps_weight=0.50,
                lambda_entropy=DEFAULT_ENTROPY_LAMBDA
            )
            w_used = weight_metadata.get('weights_used', {})
            pit_penalty_active = any(v > 0 for v in weight_metadata.get('pit_penalty_applied', {}).values())
            pit_indicator = " +PIT_penalty" if pit_penalty_active else ""
            _log(f"     ⚠️  Hyvärinen disabled (n={n_samples} < {MIN_HYVARINEN_SAMPLES}) → BIC+CRPS (bic={w_used.get('bic', 0):.2f}, crps={w_used.get('crps', 0):.2f}){pit_indicator}")
        else:
            # Full regime-aware method: BIC + Hyvärinen + CRPS + PIT
            raw_weights, weight_metadata = compute_regime_aware_model_weights(
                bic_values, hyvarinen_scores, crps_values,
                pit_pvalues=pit_pvalues,  # February 2026 - Elite PIT calibration
                berk_pvalues=berk_pvalues,
                ad_pvalues=ad_pvalues_regime,  # AD veto gate
                berkowitz_lr_stats=berk_lr_regime, pit_counts=pit_count_regime,
                mad_values=mad_values,
                regime=regime, lambda_entropy=DEFAULT_ENTROPY_LAMBDA
            )
            w_used = weight_metadata.get('weights_used', {})
            pit_penalty_active = any(v > 0 for v in weight_metadata.get('pit_penalty_applied', {}).values())
            pit_indicator = " +PIT_penalty" if pit_penalty_active else ""
            _log(f"     → Using regime-aware BIC+Hyvärinen+CRPS selection (regime={regime}, bic={w_used.get('bic', 0):.2f}, hyv={w_used.get('hyvarinen', 0):.2f}, crps={w_used.get('crps', 0):.2f}){pit_indicator}")
        
        # Store combined_score and entropy-regularized weights in each model
        for m in models:
            w = raw_weights.get(m, 1e-10)
            if weight_metadata is not None:
                # Use standardized combined score (lower = better)
                # Handle None values from metadata (non-finite scores stored as None)
                combined_score_val = weight_metadata.get('combined_scores_standardized', {}).get(m)
                models[m]['combined_score'] = float(combined_score_val) if combined_score_val is not None else 0.0
                models[m]['model_weight_entropy'] = float(w)
                bic_std_val = weight_metadata.get('bic_standardized', {}).get(m)
                models[m]['standardized_bic'] = float(bic_std_val) if bic_std_val is not None else None
                hyv_std_val = weight_metadata.get('hyvarinen_standardized', {}).get(m)
                models[m]['standardized_hyvarinen'] = float(hyv_std_val) if hyv_std_val is not None else None
                # CRPS standardized (February 2026 - regime-aware scoring)
                crps_std_val = weight_metadata.get('crps_standardized', {}).get(m)
                models[m]['standardized_crps'] = float(crps_std_val) if crps_std_val is not None else None
                # Store scoring weights used for this model
                scoring_weights = weight_metadata.get('weights_used', {})
                models[m]['scoring_weights'] = {
                    'bic': float(scoring_weights.get('bic', 0.0)),
                    'hyvarinen': float(scoring_weights.get('hyvarinen', 0.0)),
                    'crps': float(scoring_weights.get('crps', 0.0)),
                }
                models[m]['crps_scoring_enabled'] = weight_metadata.get('crps_enabled', False)
                models[m]['entropy_lambda'] = DEFAULT_ENTROPY_LAMBDA
            else:
                # Legacy: log of weight
                models[m]['combined_score'] = float(np.log(w)) if w > 0 else float('-inf')
        
        # =====================================================================
        # Step 3a: Apply Elite Tuning Fragility Penalties (v2.0 - February 2026)
        # =====================================================================
        # TOP 0.001% UPGRADE: Fragile models are down-weighted
        # 
        # CORE DESIGN CONSTRAINT: Fragility must only act as a PENALTY, not reward.
        # - Basin optimum (fragility < 0.3) → neutral (no effect)
        # - Moderate fragility (0.3-0.5) → mild penalty (10-30%)
        # - Ridge optimum (fragility > 0.5) → significant penalty (30-70%)
        #
        # This ensures BIC/Hyvarinen selection is STABILITY-AWARE, not just fit-aware.
        # =====================================================================
        weights_pre_elite = raw_weights.copy()
        elite_penalty_applied = False
        
        if ELITE_TUNING_AVAILABLE and ELITE_TUNING_ENABLED:
            for m in models:
                if not models[m].get('fit_success', False):
                    continue
                    
                elite_diag = models[m].get('diagnostics', {}).get('elite_diagnostics', {})
                if not elite_diag:
                    continue
                
                fragility = elite_diag.get('fragility_index', 0.0)
                is_ridge = elite_diag.get('is_ridge_optimum', False)
                basin_score = elite_diag.get('basin_score', 1.0)
                
                # Compute fragility penalty (asymmetric: only penalize, never reward)
                # Penalty = 0 for fragility < 0.3, then scales up
                if fragility > 0.3:
                    # Penalty scales from 0 at 0.3 to 0.7 at 1.0
                    penalty_factor = min((fragility - 0.3) / 0.7, 1.0)
                    weight_multiplier = 1.0 - (0.7 * penalty_factor)  # 1.0 → 0.3
                    
                    # Extra penalty for ridge optima (dangerous)
                    if is_ridge:
                        weight_multiplier *= 0.5  # Additional 50% penalty
                        _log(f"     ⚠️  {m}: RIDGE optimum detected (basin={basin_score:.2f}) → extra penalty")
                    
                    old_weight = raw_weights.get(m, 0.0)
                    new_weight = old_weight * weight_multiplier
                    raw_weights[m] = new_weight
                    
                    # Store penalty info
                    models[m]['elite_fragility_penalty'] = 1.0 - weight_multiplier
                    models[m]['elite_weight_pre_penalty'] = old_weight
                    models[m]['elite_weight_post_penalty'] = new_weight
                    elite_penalty_applied = True
                    
                    _log(f"     → {m}: fragility={fragility:.2f} → penalty {(1.0-weight_multiplier)*100:.0f}%")
                else:
                    models[m]['elite_fragility_penalty'] = 0.0
        
        if elite_penalty_applied:
            _log(f"     → Elite tuning penalties applied (fragility-aware BIC/Hyvärinen)")
        
        # =====================================================================
        # Step 3b: Apply Asymmetric PIT Violation Penalties (February 2026)
        # =====================================================================
        # CORE DESIGN CONSTRAINT: PIT must only act as a PENALTY, never a reward.
        # Good PIT → neutral (P=1.0, no effect)
        # Bad PIT → penalized (P<1.0, model demoted)
        # =====================================================================
        pit_penalty_report = None
        weights_pre_pit = raw_weights.copy()
        
        if PIT_PENALTY_AVAILABLE:
            # Extract PIT p-values for each model
            model_pit_pvalues = {
                m: models[m].get('pit_ks_pvalue') for m in models
            }
            
            # Apply asymmetric PIT penalties
            raw_weights, pit_penalty_report = apply_pit_penalties_to_weights(
                raw_weights=raw_weights,
                model_pit_pvalues=model_pit_pvalues,
                regime=regime,
                n_samples=n_samples,
            )
            
            # Store PIT penalty info in each model
            for m in models:
                if pit_penalty_report and m in pit_penalty_report.model_penalties:
                    penalty_result = pit_penalty_report.model_penalties[m]
                    models[m]['pit_violation_severity'] = float(penalty_result.violation_severity)
                    models[m]['pit_penalty_raw'] = float(penalty_result.raw_penalty)
                    models[m]['pit_penalty_effective'] = float(penalty_result.effective_penalty)
                    models[m]['pit_triggers_exit'] = penalty_result.triggers_exit
                    models[m]['model_weight_pre_pit'] = float(weights_pre_pit.get(m, 0.0))
                    models[m]['model_weight_post_pit'] = float(raw_weights.get(m, 0.0))
            
            # Log if PIT penalty changed model selection
            if pit_penalty_report and pit_penalty_report.selection_diverged:
                _log(f"     ⚠️  PIT penalty changed selection: {pit_penalty_report.best_model_by_fit} → {pit_penalty_report.best_model_after_penalty}")
            
            # Count and log violations
            if pit_penalty_report and pit_penalty_report.n_violated > 0:
                _log(f"     → PIT violations: {pit_penalty_report.n_violated} models penalized")
        
        # =====================================================================
        # Step 4: Apply temporal smoothing
        # =====================================================================
        smoothed_weights = apply_temporal_smoothing(raw_weights, prev_posterior, temporal_alpha)
        
        # =====================================================================
        # Step 5: Normalize to get posterior p(m|r)
        # =====================================================================
        model_posterior = normalize_weights(smoothed_weights)
        
        # Print posterior
        posterior_str = ", ".join([f"{m}={p:.3f}" for m, p in model_posterior.items()])
        _log(f"     → Posterior: {posterior_str}")
        
        # =====================================================================
        # Build regime result
        # =====================================================================
        # Compute best scores for metadata
        finite_bics = [b for b in bic_values.values() if np.isfinite(b)]
        finite_hyvs = [h for h in hyvarinen_scores.values() if np.isfinite(h)]
        finite_combined = [models[m].get('combined_score', float('inf')) for m in models if np.isfinite(models[m].get('combined_score', float('inf')))]
        
        # Best model by combined score (lowest = best for standardized scores)
        best_model_by_combined = min(models.items(), key=lambda kv: kv[1].get('combined_score', float('inf')))[0] if models else None
        
        # Build PIT penalty metadata
        pit_penalty_meta = None
        if pit_penalty_report is not None:
            pit_penalty_meta = {
                "n_violated": pit_penalty_report.n_violated,
                "n_exit_triggered": pit_penalty_report.n_exit_triggered,
                "selection_diverged": pit_penalty_report.selection_diverged,
                "best_model_by_fit": pit_penalty_report.best_model_by_fit,
                "best_model_after_penalty": pit_penalty_report.best_model_after_penalty,
                "max_penalty_model": pit_penalty_report.max_penalty_model,
                "max_penalty_value": float(pit_penalty_report.max_penalty_value),
            }
        
        regime_results[regime] = {
            "model_posterior": model_posterior,
            "models": models,
            "regime_meta": {
                "temporal_alpha": temporal_alpha,
                "n_samples": n_samples,
                "regime_name": regime_name,
                "fallback": False,
                "borrowed_from_global": False,
                "bic_min": float(min(finite_bics)) if finite_bics else None,
                "hyvarinen_max": float(max(finite_hyvs)) if finite_hyvs else None,
                "combined_score_min": float(min(finite_combined)) if finite_combined else None,
                "best_model_by_combined": best_model_by_combined,
                "model_selection_method": "regime_aware_crps",
                "effective_selection_method": "bic_crps_only" if hyvarinen_disabled else "bic_hyv_crps",
                "hyvarinen_disabled": hyvarinen_disabled,
                "crps_enabled": True,
                "entropy_lambda": DEFAULT_ENTROPY_LAMBDA if model_selection_method == 'combined' else None,
                "smoothing_applied": prev_posterior is not None and temporal_alpha > 0,
                # PIT Penalty metadata (February 2026)
                "pit_penalty_applied": pit_penalty_report is not None,
                "pit_penalty": pit_penalty_meta,
                # Elite Tuning metadata (v2.0 - February 2026)
                "elite_tuning_enabled": ELITE_TUNING_AVAILABLE and ELITE_TUNING_ENABLED,
                "elite_tuning_preset": ELITE_TUNING_PRESET if ELITE_TUNING_AVAILABLE else None,
                "elite_penalty_applied": elite_penalty_applied,
            }
        }
    
    return regime_results


def tune_regime_model_averaging(
    returns: np.ndarray,
    vol: np.ndarray,
    regime_labels: np.ndarray,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
    min_samples: int = MIN_REGIME_SAMPLES,
    temporal_alpha: float = DEFAULT_TEMPORAL_ALPHA,
    previous_posteriors: Optional[Dict[int, Dict[str, float]]] = None,
    lambda_regime: float = 0.05,
    model_selection_method: str = DEFAULT_MODEL_SELECTION_METHOD,
    bic_weight: float = DEFAULT_BIC_WEIGHT,
    prices: np.ndarray = None,  # Added for MR integration (February 2026)
    asset: str = None,  # FIX #4: Asset symbol for c-bounds detection
    gk_c_prior_value: float = None,  # Story 2.2: GK-informed c prior
) -> Dict:
    """
    Full regime-conditional Bayesian model averaging pipeline.
    
    This is the main entry point for the upgraded tuning system.
    It combines:
    1. Global model fitting (fallback)
    2. Regime-conditional model fitting with BMA
    3. Temporal smoothing of model posteriors
    4. Hierarchical shrinkage toward global
    5. Robust model selection via Hyvärinen score (optional)
    
    Args:
        returns: Array of returns
        vol: Array of EWMA volatility
        regime_labels: Array of regime labels (0-4)
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        min_samples: Minimum samples per regime
        temporal_alpha: Smoothing exponent for model posteriors
        previous_posteriors: Previous model posteriors per regime (for smoothing)
        lambda_regime: Hierarchical shrinkage strength
        model_selection_method: 'bic', 'hyvarinen', or 'combined' (default)
        bic_weight: Weight for BIC in combined method (0-1, default 0.5)
        
    Returns:
        Dictionary with:
        {
            "global": { global model fits },
            "regime": {
                r: {
                    "model_posterior": { m: p(m|r) },
                    "models": { m: {...} },
                    "regime_meta": {...}
                }
            },
            "meta": {
                "temporal_alpha": ...,
                "lambda_regime": ...,
                ...
            }
        }
    """
    # Validate inputs
    returns = np.asarray(returns).flatten()
    vol = np.asarray(vol).flatten()
    regime_labels = np.asarray(regime_labels).flatten().astype(int)
    
    n_obs = len(returns)
    _log(f"  📊 Bayesian Model Averaging: {n_obs} observations, α={temporal_alpha:.2f}")
    _log(f"  📊 Model selection method: {model_selection_method}" + 
         (f" (BIC weight={bic_weight:.2f})" if model_selection_method == 'combined' else ""))
    
    # =========================================================================
    # Step 1: Fit global models (fallback)
    # =========================================================================
    _log(f"  🔧 Fitting global models...")
    global_models = fit_all_models_for_regime(
        returns, vol,
        prior_log_q_mean=prior_log_q_mean,
        prior_lambda=prior_lambda,
        prices=prices,  # MR integration (February 2026)
        regime_labels=regime_labels,
        asset=asset,  # FIX #4: Asset-class adaptive c bounds
        gk_c_prior_value=gk_c_prior_value,  # Story 2.2
    )

    # Apply global q floor (Story 1.1): use regime-count-weighted average floor
    # This ensures global fallback models also have meaningful drift sensitivity
    if regime_labels is not None and len(regime_labels) > 0:
        _regime_counts = np.bincount(regime_labels.astype(int), minlength=5)
        _regime_total = max(_regime_counts.sum(), 1)
        _global_q_floor = sum(
            Q_FLOOR_BY_REGIME.get(r, 0) * _regime_counts[r] / _regime_total
            for r in range(5)
        )
    else:
        # No regime info: use median floor
        _global_q_floor = 5e-5  # median of Q_FLOOR_BY_REGIME values

    _n_fl_global, _n_total_global = 0, 0
    for _gm_name, _gm_info in global_models.items():
        if not _gm_info.get("fit_success", False):
            continue
        _n_total_global += 1
        _gm_q = _gm_info.get("q")
        if _gm_q is not None and _gm_q < _global_q_floor:
            _n_fl_global += 1
            _gm_info["q_mle_original"] = float(_gm_q)
            _gm_info["q"] = float(_global_q_floor)
            _gm_info["q_floor_applied"] = True
            _gm_info["q_floor_value"] = float(_global_q_floor)
        else:
            _gm_info["q_floor_applied"] = False

    if _n_fl_global > 0:
        _log(f"     Global Q-floor applied: {_n_fl_global}/{_n_total_global} models "
             f"(weighted floor={_global_q_floor:.1e})")

    # Compute global model posterior using elite CRPS-dominated scoring
    global_bic = {m: global_models[m].get("bic", float('inf')) for m in global_models if global_models[m].get("fit_success", False)}
    global_hyvarinen = {m: global_models[m].get("hyvarinen_score", float('-inf')) for m in global_models if global_models[m].get("fit_success", False)}
    global_crps = {m: global_models[m]["crps"] for m in global_models
                   if global_models[m].get("fit_success", False) and global_models[m].get("crps") is not None and np.isfinite(global_models[m]["crps"])}
    global_pit = {m: global_models[m]["pit_ks_pvalue"] for m in global_models
                  if global_models[m].get("fit_success", False) and global_models[m].get("pit_ks_pvalue") is not None}
    global_berk = {m: global_models[m]["berkowitz_pvalue"] for m in global_models
                   if global_models[m].get("fit_success", False) and global_models[m].get("berkowitz_pvalue") is not None}
    global_berk_lr = {m: global_models[m]["berkowitz_lr"] for m in global_models
                      if global_models[m].get("fit_success", False) and global_models[m].get("berkowitz_lr") is not None}
    global_pit_counts = {m: global_models[m]["pit_count"] for m in global_models
                         if global_models[m].get("fit_success", False) and global_models[m].get("pit_count") is not None}
    global_mad = {m: global_models[m]["histogram_mad"] for m in global_models
                  if global_models[m].get("fit_success", False) and global_models[m].get("histogram_mad") is not None}
    global_ad = {m: global_models[m]["ad_pvalue"] for m in global_models
                 if global_models[m].get("fit_success", False) and global_models[m].get("ad_pvalue") is not None
                 and np.isfinite(global_models[m]["ad_pvalue"])}
    fallback_weight_metadata = None
    
    if global_crps and CRPS_SCORING_ENABLED:
        global_raw_weights, fallback_weight_metadata = compute_regime_aware_model_weights(
            global_bic, global_hyvarinen, global_crps,
            pit_pvalues=global_pit, berk_pvalues=global_berk,
            ad_pvalues=global_ad,  # AD veto gate
            berkowitz_lr_stats=global_berk_lr, pit_counts=global_pit_counts,
            mad_values=global_mad, regime=None,
            lambda_entropy=DEFAULT_ENTROPY_LAMBDA
        )
    elif model_selection_method == 'bic':
        # Use vectorized BMA weights when available (log-sum-exp stable)
        if VECTORIZED_OPS_AVAILABLE and global_bic:
            model_names = list(global_bic.keys())
            bic_arr = np.array([global_bic[m] for m in model_names])
            w_arr = vectorized_bma_weights(bic_arr)
            global_raw_weights = dict(zip(model_names, w_arr.tolist()))
        else:
            global_raw_weights = compute_bic_model_weights(global_bic)
    elif model_selection_method == 'hyvarinen':
        global_raw_weights = compute_hyvarinen_model_weights(global_hyvarinen)
    else:
        global_raw_weights, fallback_weight_metadata = compute_combined_model_weights(
            global_bic, global_hyvarinen, bic_weight=bic_weight,
            lambda_entropy=DEFAULT_ENTROPY_LAMBDA
        )
    
    # Store combined_score and entropy-regularized weights in each global model
    for m in global_models:
        w = global_raw_weights.get(m, 1e-10)
        if fallback_weight_metadata is not None:
            combined_score_val = fallback_weight_metadata.get('combined_scores_standardized', {}).get(m)
            global_models[m]['combined_score'] = float(combined_score_val) if combined_score_val is not None else 0.0
            global_models[m]['model_weight_entropy'] = float(w)
            bic_std_val = fallback_weight_metadata.get('bic_standardized', {}).get(m)
            global_models[m]['standardized_bic'] = float(bic_std_val) if bic_std_val is not None else None
            hyv_std_val = fallback_weight_metadata.get('hyvarinen_standardized', {}).get(m)
            global_models[m]['standardized_hyvarinen'] = float(hyv_std_val) if hyv_std_val is not None else None
            crps_std_val = fallback_weight_metadata.get('crps_standardized', {}).get(m)
            global_models[m]['standardized_crps'] = float(crps_std_val) if crps_std_val is not None else None
            scoring_weights = fallback_weight_metadata.get('weights_used', {})
            global_models[m]['scoring_weights'] = dict(scoring_weights)
            global_models[m]['crps_scoring_enabled'] = fallback_weight_metadata.get('crps_enabled', False)
            global_models[m]['entropy_lambda'] = DEFAULT_ENTROPY_LAMBDA
        else:
            global_models[m]['combined_score'] = float(np.log(w)) if w > 0 else float('-inf')
    
    global_posterior = normalize_weights(global_raw_weights)
    
    _log(f"     Global posterior: " + ", ".join([f"{m}={p:.3f}" for m, p in global_posterior.items()]))

    # =========================================================================
    # Alternative BMA weight computations (stored alongside primary weights)
    # =========================================================================

    # Story 4.1: LOO-CRPS model evaluation (per-observation scoring)
    # Uses vectorized Numba functions (arrays in, array out), not scalar-per-step.
    _crps_arrays = {}  # model_name -> ndarray(T,) for stacking matrix
    if LOO_CRPS_AVAILABLE and global_models:
        try:
            from models.gaussian import kalman_filter_drift_phi
            _loo_crps_scores = {}
            for _m_name, _m_info in global_models.items():
                if not _m_info.get("fit_success", False):
                    continue
                _mq = _m_info.get("q", 1e-5)
                _mc = _m_info.get("c", 1.0)
                _mp = _m_info.get("phi", 0.0)
                _mn = _m_info.get("nu")
                _mu_arr, _P_arr, _ = kalman_filter_drift_phi(returns, vol, _mq, _mc, _mp)
                _sigma_arr = np.sqrt(_P_arr + _mc * vol ** 2)
                # Vectorized LOO-CRPS: pass full arrays, get array back
                if _mn is not None:
                    _nu_arr = np.full(len(returns), float(_mn))
                    _crps_per_obs = loo_crps_student_t(_mu_arr, _sigma_arr, _nu_arr, returns)
                else:
                    _crps_per_obs = loo_crps_gaussian(_mu_arr, _sigma_arr, returns)
                # Skip first observation (filter needs burn-in)
                _valid = _crps_per_obs[1:]
                _finite_mask = np.isfinite(_valid)
                if _finite_mask.sum() > 0:
                    _loo_crps_scores[_m_name] = float(np.mean(_valid[_finite_mask]))
                    _crps_arrays[_m_name] = _crps_per_obs  # Full array for stacking
            if _loo_crps_scores:
                for _m_name in global_models:
                    if _m_name in _loo_crps_scores:
                        global_models[_m_name]["loo_crps"] = float(_loo_crps_scores[_m_name])
        except Exception:
            pass

    # Story 4.2: CRPS stacking with proper (T, M) matrix
    # Story 4.3: Temporal CRPS stacking with exponential forgetting
    if CRPS_STACKING_AVAILABLE and global_models and _crps_arrays:
        try:
            _model_names_cs = [m for m in _crps_arrays if m in global_models]
            if len(_model_names_cs) >= 2:
                # Build proper (T, M) CRPS matrix from per-observation arrays
                _T = len(returns)
                _M = len(_model_names_cs)
                _crps_matrix = np.ones((_T, _M))
                for _i, _m_name in enumerate(_model_names_cs):
                    _crps_matrix[:, _i] = _crps_arrays[_m_name]
                # BIC weights as warm start
                _bic_w = np.array([global_raw_weights.get(m, 1e-10) for m in _model_names_cs])
                _bic_w = _bic_w / _bic_w.sum()

                # Story 4.2: Static CRPS stacking
                _crps_stack = crps_stacking_weights(_crps_matrix, bic_weights=_bic_w)
                _stack_w = _crps_stack.weights if hasattr(_crps_stack, 'weights') else _crps_stack
                for _i, _m_name in enumerate(_model_names_cs):
                    if hasattr(_stack_w, '__getitem__') and _i < len(_stack_w):
                        global_models[_m_name]["crps_stacking_weight"] = float(_stack_w[_i])

                # Story 4.3: Temporal CRPS stacking (diagnostic)
                try:
                    _temp_stack = temporal_crps_stacking(
                        _crps_matrix, bic_weights=_bic_w, lambda_decay=0.995,
                    )
                    _tw = _temp_stack.weights if hasattr(_temp_stack, 'weights') else _temp_stack
                    for _i, _m_name in enumerate(_model_names_cs):
                        if hasattr(_tw, '__getitem__') and _i < len(_tw):
                            global_models[_m_name]["temporal_crps_weight"] = float(_tw[_i])
                except Exception:
                    pass
        except Exception:
            pass

    # Story 6.1: Entropy-regularized BMA
    if ENTROPY_BMA_AVAILABLE and global_bic:
        try:
            _ebma_names = list(global_bic.keys())
            _ebma_lls = np.array([-0.5 * global_bic[m] for m in _ebma_names])  # Approx LL from BIC
            _ebma_nparams = np.array([global_models.get(m, {}).get('n_params', 3) for m in _ebma_names])
            _ebma = entropy_regularized_bma(_ebma_lls, _ebma_nparams, len(returns))
            _ebma_w = _ebma.weights if hasattr(_ebma, 'weights') else _ebma
            for _i, _m_name in enumerate(_ebma_names):
                if _i < len(_ebma_w):
                    global_models[_m_name]["entropy_bma_weight"] = float(_ebma_w[_i])
        except Exception:
            pass

    # Story 6.2: MDL weights (diagnostic comparison to BIC)
    if MDL_WEIGHTS_AVAILABLE and global_bic:
        try:
            _mdl_names = list(global_bic.keys())
            _mdl_lls = np.array([-0.5 * global_bic[m] for m in _mdl_names])
            _mdl_nparams = np.array([global_models.get(m, {}).get('n_params', 3) for m in _mdl_names])
            _mdl = mdl_weights(_mdl_lls, _mdl_nparams, len(returns))
            _mdl_w = _mdl.weights if hasattr(_mdl, 'weights') else _mdl
            for _i, _m_name in enumerate(_mdl_names):
                if _i < len(_mdl_w):
                    global_models[_m_name]["mdl_weight"] = float(_mdl_w[_i])
        except Exception:
            pass

    # =========================================================================
    # Step 2: Fit regime-conditional models with BMA
    # =========================================================================
    _log(f"  🔄 Fitting regime-conditional models...")
    regime_results = fit_regime_model_posterior(
        returns, vol, regime_labels,
        prior_log_q_mean=prior_log_q_mean,
        prior_lambda=prior_lambda,
        min_samples=MIN_REGIME_SAMPLES,
        temporal_alpha=DEFAULT_TEMPORAL_ALPHA,
        previous_posteriors=previous_posteriors,
        global_models=global_models,
        global_posterior=global_posterior,
        model_selection_method=model_selection_method,
        bic_weight=bic_weight,
        prices=prices,  # MR integration (February 2026)
        asset=asset,  # FIX #4: Asset-class adaptive c bounds
    )
    
    # =========================================================================
    # Step 3: Apply hierarchical shrinkage to regime posteriors (optional)
    # =========================================================================
    if lambda_regime > 0:
        _log(f"  📐 Applying hierarchical shrinkage (λ={lambda_regime:.3f})...")
        for r, r_result in regime_results.items():
            if r_result.get("regime_meta", {}).get("fallback", False):
                continue
            
            n_samples = r_result.get("regime_meta", {}).get("n_samples", 0)
            if n_samples < min_samples:
                continue
            
            # Shrinkage factor
            sf = 1.0 / (1.0 + lambda_regime * min_samples / max(n_samples, 1.0))
            
            # Shrink model posteriors toward global
            shrunk_posterior = {}
            for m in r_result["model_posterior"]:
                p_regime = r_result["model_posterior"][m]
                p_global = global_posterior.get(m, 1.0 / 3.0)
                p_shrunk = sf * p_regime + (1 - sf) * p_global
                shrunk_posterior[m] = p_shrunk
            
            # Renormalize
            shrunk_posterior = normalize_weights(shrunk_posterior)
            r_result["model_posterior_unshrunk"] = r_result["model_posterior"]
            r_result["model_posterior"] = shrunk_posterior
            r_result["regime_meta"]["shrinkage_applied"] = True
            r_result["regime_meta"]["shrinkage_factor"] = float(sf)
    
    # =========================================================================
    # Build final result
    # =========================================================================
    # Compute global Hyvärinen max for metadata
    global_hyvarinen_scores = [
        global_models[m].get("hyvarinen_score", float('-inf')) 
        for m in global_models 
        if global_models[m].get("fit_success", False)
    ]
    global_hyvarinen_max = max(global_hyvarinen_scores) if global_hyvarinen_scores else None
    
    global_bic_scores = [
        global_models[m].get("bic", float('inf')) 
        for m in global_models 
        if global_models[m].get("fit_success", False)
    ]
    global_bic_min = min(global_bic_scores) if global_bic_scores else None
    
    # Compute global combined_score_min for metadata (lower = better for standardized scores)
    global_combined_scores = [
        global_models[m].get("combined_score", float('inf')) 
        for m in global_models 
        if global_models[m].get("fit_success", False) and np.isfinite(global_models[m].get("combined_score", float('inf')))
    ]
    global_combined_score_min = min(global_combined_scores) if global_combined_scores else None
    
    result = {
        "global": {
            "model_posterior": global_posterior,
            "models": global_models,
            "hyvarinen_max": float(global_hyvarinen_max) if global_hyvarinen_max is not None and np.isfinite(global_hyvarinen_max) else None,
            "combined_score_min": float(global_combined_score_min) if global_combined_score_min is not None and np.isfinite(global_combined_score_min) else None,
            "bic_min": float(global_bic_min) if global_bic_min is not None and np.isfinite(global_bic_min) else None,
            "model_selection_method": model_selection_method,
            "bic_weight": bic_weight if model_selection_method == 'combined' else None,
            "entropy_lambda": DEFAULT_ENTROPY_LAMBDA if model_selection_method == 'combined' else None,
            # Elite Tuning metadata (v2.0 - February 2026)
            "elite_tuning_enabled": ELITE_TUNING_AVAILABLE and ELITE_TUNING_ENABLED,
            "elite_tuning_preset": ELITE_TUNING_PRESET if ELITE_TUNING_AVAILABLE else None,
        },
        "regime": regime_results,
        "meta": {
            "temporal_alpha": temporal_alpha,
            "lambda_regime": lambda_regime,
            "n_obs": n_obs,
            "min_samples": min_samples,
            "model_selection_method": model_selection_method,
            "bic_weight": bic_weight if model_selection_method == 'combined' else None,
            "entropy_lambda": DEFAULT_ENTROPY_LAMBDA if model_selection_method == 'combined' else None,
            "n_regimes_active": sum(1 for r in regime_results.values() 
                                    if not r.get("regime_meta", {}).get("fallback", False)),
            # Elite Tuning configuration (v2.0 - February 2026)
            "elite_tuning_available": ELITE_TUNING_AVAILABLE,
            "elite_tuning_enabled": ELITE_TUNING_ENABLED,
            "elite_tuning_preset": ELITE_TUNING_PRESET if ELITE_TUNING_AVAILABLE else None,
        }
    }
    
    return result


