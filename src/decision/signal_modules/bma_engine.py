from __future__ import annotations

"""Bayesian Model Averaging MC engine for signal generation.

Extracted from signals.py - Story 7.2.
Contains: bayesian_model_average_mc.
"""

import math
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys as _sys
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

from decision.signal_modules.config import *  # noqa: F403
from decision.signal_modules.volatility_imports import *  # noqa: F403
from decision.signal_modules.regime_classification import *  # noqa: F403
from decision.signal_modules.monte_carlo import *  # noqa: F403

def bayesian_model_average_mc(
    feats: Dict[str, pd.Series],
    regime_params: Dict[int, Dict],
    mu_t: float,
    P_t: float,
    sigma2_step: float,
    H: int,
    n_paths: int = 10000,
    seed: Optional[int] = None,
    tuned_params: Optional[Dict] = None,
    asset_symbol: Optional[str] = None,
    horizons_extract: Optional[List[int]] = None,
    phi_slow: float = 0.0,
    mu_slow_0: float = 0.0,
    # Story 3.4: Asset-class-aware per-step return cap
    return_cap: float = 0.30,
    # Story 4.5: Previous smoothed regime probs for EMA smoothing
    prev_regime_probs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[int, float], Dict]:
    """
    Perform Bayesian Model Averaging using CURRENT REGIME's model posterior.

    Implements the posterior predictive for current regime r_t:

        p(x | D, r_t) = Σ_m p(x | r_t, m, θ_{r_t,m}) · p(m | r_t)

    KEY DESIGN:
    - Determines CURRENT regime using same logic as tune.py (deterministic)
    - Uses that regime's model_posterior and models from tuning
    - Does NOT blend across regimes - uses single regime's BMA
    - Falls back to global if regime data unavailable

    SOFT REGIME PROBABILITIES (for Trust Authority):
    - Hard regime assignment remains for parameter selection
    - Soft probabilities computed for trust modulation to avoid cliffs
    - Based on regime transition smoothing: current=0.7, adjacent=0.15 each

    CONTRACT WITH tune.py:
    - Regime assignment uses SAME logic as assign_regime_labels()
    - Every regime contains model_posterior and models (even fallbacks)
    - Model posteriors are already temporally smoothed by tune
    - This function does NOT perform tuning, BIC/AIC, or temporal smoothing

    CRITICAL RULES:
    - Do NOT perform tuning here
    - Do NOT recompute likelihoods
    - Do NOT renormalize model weights (already normalized by tune)
    - Do NOT apply temporal smoothing to model posteriors
    - Do NOT select best model - use full BMA mixture
    - Do NOT synthesize fake models - use hierarchical fallback to global

    Args:
        feats: Feature dictionary with 'ret', 'vol', etc.
        regime_params: (Legacy - ignored if tuned_params provided)
        mu_t: Current drift estimate
        P_t: Current drift variance
        sigma2_step: Per-step volatility
        H: Forecast horizon
        n_paths: Total MC paths to generate
        seed: Random seed for reproducibility
        tuned_params: Full tuned params from _load_tuned_kalman_params (BMA structure)

    Returns:
        Tuple of:
        - r_samples: Samples from posterior predictive p(x | D, r_t)
        - regime_probs: Soft probability dict for trust authority {regime_idx: prob}
        - metadata: Diagnostic information
    """
    rng = np.random.default_rng(seed)

    # Check if we have new BMA structure with model posteriors
    has_bma = tuned_params is not None and tuned_params.get('has_bma', False)

    # If no BMA structure available, this is old cache format - REJECT
    if not has_bma:
        # Determine the reason for missing BMA structure
        if tuned_params is None:
            reason = "tuned_params is None (asset not in cache/tune/)"
            error_type = "MISSING_CACHE"
        elif 'global' not in tuned_params:
            reason = "missing 'global' key (old cache format)"
            error_type = "OLD_CACHE_FORMAT"
        else:
            reason = "has_bma is False or missing"
            error_type = "BMA_DISABLED"
        
        # Format asset display name
        asset_display = asset_symbol if asset_symbol else "UNKNOWN"
        
        # Print improved warning with asset name
        print(f"\n⚠️  BMA STRUCTURE MISSING for {asset_display}", file=sys.stderr)
        print(f"   ├─ Reason: {reason}", file=sys.stderr)
        print(f"   ├─ Impact: Signals will show 0% for all horizons", file=sys.stderr)
        print(f"   └─ Fix: Run 'make tune --assets {asset_display}' to regenerate cache\n", file=sys.stderr)
        
        # Save failure to failed_assets.json for tracking
        try:
            if asset_symbol:
                failure_info = {
                    asset_symbol: {
                        "display_name": asset_symbol,
                        "attempts": 1,
                        "last_error": f"BMA structure missing: {reason}",
                        "error_type": error_type,
                        "traceback": None,
                    }
                }
                save_failed_assets(failure_info, append=True)
        except Exception:
            pass  # Don't fail signal generation due to logging error

        # Return uniform soft regime probs for trust (maximally uncertain)
        uniform_regime_probs = {i: 0.2 for i in range(5)}
        return np.array([0.0]), np.array([0.0]), uniform_regime_probs, {
            "method": "REJECTED",
            "reason": "no_bma_structure_old_cache_format",
            "error": "Cache must be regenerated with tune.py for BMA support",
            "debug_reason": reason,
            "asset": asset_display,
            "error_type": error_type,
        }

    # ========================================================================
    # DETERMINE CURRENT REGIME (DETERMINISTIC - SAME AS TUNE)
    # ========================================================================
    # Use exact same logic as tune.py assign_regime_labels()
    # This ensures consistency between tuning and inference
    # ========================================================================
    current_regime = assign_current_regime(feats)
    regime_name = REGIME_NAMES.get(current_regime, f"REGIME_{current_regime}")

    # ========================================================================
    # Story 4.1: SOFT REGIME PROBABILITIES VIA LOGISTIC BOUNDARIES
    # ========================================================================
    # Compute v2 probabilistic regime assignment using logistic transitions.
    # This replaces the fixed 0.7/0.3 adjacency heuristic.
    # ========================================================================
    # Extract vol_relative, drift_abs, tail_indicator from feats
    _vol_series = feats.get("vol", pd.Series(dtype=float))
    _ret_series = feats.get("ret", pd.Series(dtype=float))
    _lookback = 21

    if isinstance(_vol_series, pd.Series) and not _vol_series.empty:
        _vol_now = float(_vol_series.iloc[-1])
        if len(_vol_series) >= _lookback:
            _vol_median = float(_vol_series.expanding(
                min_periods=min(_lookback, len(_vol_series))).median().iloc[-1])
        else:
            _vol_median = float(_vol_series.median())
        _vol_relative = _vol_now / _vol_median if _vol_median > 1e-12 else 1.0
    else:
        _vol_relative = 1.0
        _vol_now = 0.0

    if isinstance(_ret_series, pd.Series) and not _ret_series.empty:
        _drift_abs = abs(float(_ret_series.tail(_lookback).mean())) if len(_ret_series) >= _lookback else abs(float(_ret_series.mean()))
        _ret_now = float(_ret_series.iloc[-1])
        _tail_indicator = abs(_ret_now) / _vol_now if _vol_now > 1e-12 else 0.0
    else:
        _drift_abs = 0.0
        _tail_indicator = 0.0

    v2_regime_probs = compute_regime_probabilities_v2(
        vol_relative=_vol_relative,
        drift_abs=_drift_abs,
        tail_indicator=_tail_indicator,
    )

    # ========================================================================
    # Story 4.5: EMA SMOOTH REGIME PROBABILITIES
    # ========================================================================
    v2_regime_probs = smooth_regime_probabilities(
        v2_regime_probs, prev_regime_probs,
    )

    # Soft regime probs dict for trust authority (backward-compatible format)
    soft_regime_probs = {i: float(v2_regime_probs[i]) for i in range(5)}

    # Legacy one-hot array for backward compatibility
    regime_probs_array = np.zeros(5)
    regime_probs_array[current_regime] = 1.0

    # ========================================================================
    # GET REGIME-SPECIFIC BMA DATA
    # ========================================================================
    global_data = tuned_params.get('global') or {} if tuned_params else {}
    regime_data = tuned_params.get('regime') or {} if tuned_params else {}
    
    # Ensure regime_data is a dict (could be None from old cache)
    if regime_data is None:
        regime_data = {}

    global_model_posterior = global_data.get('model_posterior', {}) if global_data else {}
    global_models = global_data.get('models', {}) if global_data else {}

    # ========================================================================
    # EXTRACT AUGMENTATION LAYER DATA
    # ========================================================================
    # When unified model is selected, external augmentation layers are disabled
    # because unified models already incorporate VoV, asymmetry, and MS-q.
    is_unified_model = global_data.get('unified_model', False) if global_data else False
    
    # Global Hansen/CST: removed (now handled per-model via Stage 7.5/7.6 and U-H/U-C)
    # Variables kept for metadata/downstream compatibility
    hansen_lambda_global = None
    hansen_nu_global = None
    hansen_skew_t_enabled = False
    cst_nu_normal_global = None
    cst_nu_crisis_global = None
    cst_epsilon_global = None
    cst_enabled = False

    # Get current regime's model_posterior and models
    regime_key = str(current_regime)  # JSON keys are strings
    r_data = regime_data.get(regime_key) or regime_data.get(current_regime)

    if r_data is not None and isinstance(r_data, dict):
        model_posterior = r_data.get('model_posterior', {})
        models = r_data.get('models', {})
        regime_meta = r_data.get('regime_meta', {})
        is_fallback = regime_meta.get('fallback', False) or regime_meta.get('borrowed_from_global', False)
    else:
        # Use global as fallback if regime data missing
        model_posterior = global_model_posterior
        models = global_models
        is_fallback = True
        regime_meta = {}

    # If models still empty, use global
    if not models or not model_posterior:
        model_posterior = global_model_posterior
        models = global_models
        is_fallback = True

    # ========================================================================
    # Story 4.1: SOFT BMA WEIGHT MIXING ACROSS REGIMES
    # ========================================================================
    # Instead of using a single regime's posteriors, mix model weights
    # across all regimes using v2 probabilistic regime assignment.
    # This eliminates abrupt weight switches at regime boundaries.
    # ========================================================================
    _soft_mixed = False
    if regime_data and global_model_posterior:
        soft_model_posterior = compute_soft_bma_weights(
            regime_probs=v2_regime_probs,
            regime_data=regime_data,
            global_model_posterior=global_model_posterior,
        )
        if soft_model_posterior:
            model_posterior = soft_model_posterior
            _soft_mixed = True
            # Collect all models from all regimes + global for parameter lookup
            _all_models = dict(global_models)
            for _rk in regime_data:
                _rb = regime_data[_rk]
                if isinstance(_rb, dict) and 'models' in _rb:
                    _all_models.update(_rb['models'])
            if _all_models:
                models = _all_models

    # If still empty after global fallback - cannot proceed
    if not models or not model_posterior:
        return np.array([0.0]), np.array([0.0]), soft_regime_probs, {
            "method": "FAILED",
            "reason": "no_models_available",
            "error": "No model posterior or models available for inference",
            "current_regime": current_regime,
            "regime_name": regime_name,
        }

    # ========================================================================
    # EPISTEMIC WEIGHTING: Use cached posteriors from tune.py when available
    # ========================================================================
    # tune.py computes posteriors through a rigorous 5-step pipeline:
    #   1. Entropy-regularized softmax (λ=0.05) over 6-component combined_score
    #   2. Elite fragility penalties (down-weight fragile/ridge models)
    #   3. PIT violation penalties (asymmetric demotions for miscalibrated models)
    #   4. Temporal smoothing with previous posterior
    #   5. Normalization
    #
    # These cached posteriors preserve ALL penalty layers. We use them directly
    # when available, falling back to recomputation ONLY when cache is missing
    # or corrupt. The recomputation now uses the correct temperature (λ=0.05)
    # matching diagnostics.py's DEFAULT_ENTROPY_LAMBDA.
    # ========================================================================
    cached_posterior_valid = (
        model_posterior
        and isinstance(model_posterior, dict)
        and len(model_posterior) > 0
        and all(isinstance(v, (int, float)) and np.isfinite(v) for v in model_posterior.values())
        and abs(sum(model_posterior.values()) - 1.0) < 0.05  # Must sum to ~1
    )

    if cached_posterior_valid:
        # Use tune.py's cached posteriors directly — preserves fragility, PIT,
        # and temporal smoothing penalties that would be lost on recomputation.
        recomputed_posterior = model_posterior
        epistemic_meta = {
            'method': 'cached_posterior',
            'reason': 'using_tune_posteriors_with_all_penalty_layers',
            'n_models': len(model_posterior),
        }
    else:
        # Fallback: recompute from combined_score with correct temperature
        recomputed_posterior, epistemic_meta = compute_model_posteriors_from_combined_score(models)

    if not recomputed_posterior:
        # Cache may be from an older version without combined_score
        # Fallback: synthesize combined_score from per-model metrics
        # This is a GRACEFUL DEGRADATION, not a hard failure
        import warnings
        model_names = list(models.keys())
        missing_scores = [m for m in model_names if not models.get(m, {}).get('combined_score')]
        
        # === METRIC SYNTHESIS FALLBACK (March 2026) ===
        # When combined_score is missing, synthesize from per-model CRPS/BIC/PIT
        # This prevents the uniform-weight bug for assets with older caches.
        synth_scores = {}
        for model_name, model_data in models.items():
            if not isinstance(model_data, dict):
                continue
            crps_val = model_data.get('crps')
            bic_val = model_data.get('bic')
            pit_p = model_data.get('pit_ks_pvalue')
            ad_p = model_data.get('ad_pvalue')
            # Need at least CRPS or BIC to synthesize a score
            if crps_val is not None and np.isfinite(crps_val):
                # Score = CRPS_component + PIT_penalty + AD_penalty  (lower = better)
                score = float(crps_val)
                if pit_p is not None and np.isfinite(pit_p) and pit_p > 0:
                    score += 0.15 * (-np.log10(max(pit_p, 1e-10)))  # PIT deviation
                if ad_p is not None and np.isfinite(ad_p) and ad_p > 0:
                    score += 0.15 * (-np.log10(max(ad_p, 1e-10)))   # AD deviation
                synth_scores[model_name] = score
            elif bic_val is not None and np.isfinite(bic_val):
                synth_scores[model_name] = float(bic_val)
        
        if synth_scores:
            # Use softmax over negated synth scores (lower = better = higher weight)
            s_names = list(synth_scores.keys())
            s_vals = np.array([synth_scores[m] for m in s_names])
            neg_s = -s_vals
            neg_s = neg_s - neg_s.max()
            weights = np.exp(neg_s)
            weights = weights / weights.sum()
            # Entropy floor
            min_w = 0.01 / max(len(s_names), 1)
            weights = np.maximum(weights, min_w)
            weights = weights / weights.sum()
            recomputed_posterior = dict(zip(s_names, weights))
            epistemic_meta = {
                'method': 'metric_synthesis_fallback',
                'reason': 'combined_score_missing',
                'components_used': ['crps', 'pit_ks_pvalue', 'ad_pvalue', 'bic'],
                'missing_scores': missing_scores,
            }
        else:
            # Try pure BIC-based fallback
            bic_posterior = {}
            for model_name, model_data in models.items():
                if isinstance(model_data, dict):
                    bic = model_data.get('bic')
                    if bic is not None and np.isfinite(bic):
                        bic_posterior[model_name] = bic
            
            if bic_posterior:
                # Convert BIC to weights using softmax over negated BIC (lower is better)
                bic_values = np.array(list(bic_posterior.values()))
                neg_bic = -bic_values / 2.0  # Standard BIC to log-likelihood conversion
                neg_bic = neg_bic - neg_bic.max()  # Numerical stability
                weights = np.exp(neg_bic)
                weights = weights / weights.sum()
                
                recomputed_posterior = dict(zip(bic_posterior.keys(), weights))
                epistemic_meta = {
                    'method': 'bic_fallback',
                    'reason': 'combined_score_missing',
                    'missing_scores': missing_scores,
                }
                
                warnings.warn(
                    f"Using BIC-based fallback for models missing combined_score: {missing_scores}. "
                    f"Consider re-tuning with 'make tune --force' for optimal calibration.",
                    RuntimeWarning
                )
            else:
                # Last resort: uniform weights
                recomputed_posterior = {m: 1.0/len(models) for m in models}
                epistemic_meta = {
                    'method': 'uniform_fallback',
                    'reason': 'no_valid_scores',
                }
                
                warnings.warn(
                    f"Using uniform weights fallback - no valid BIC or combined_score found. "
                    f"Re-tune with 'make tune --force'.",
                    RuntimeWarning
                )

    cached_posterior = model_posterior
    model_posterior = recomputed_posterior
    posteriors_recomputed = True

    # ========================================================================
    # FAIL-FAST ASSERTION: Validate model synchronization
    # ========================================================================
    # This is the #1 silent failure mode in production quant systems:
    # tune.py adds a model but signals.py ignores it → distorted posterior mass.
    #
    # The model registry (models/model_registry.py) provides the canonical
    # contract. If registry is available, we validate against it.
    #
    # ARCHITECTURAL LAW: Top funds REFUSE TO TRADE without this assertion.
    # ========================================================================
    if MODEL_REGISTRY_AVAILABLE:
        try:
            tuned_model_names = set(models.keys())
            # Only warn, don't fail - allows for temporary model additions
            # during experimentation while still flagging the issue
            assert_models_synchronised(
                tuned_model_names, 
                context=f"regime={current_regime} ({regime_name}), asset={asset_symbol}"
            )
        except AssertionError as sync_error:
            # Log but don't fail - degraded operation is better than crash
            # This will be logged to stderr for monitoring
            import warnings
            warnings.warn(
                f"Model synchronization warning: {sync_error}",
                RuntimeWarning
            )
        except Exception:
            pass  # Registry check failed - continue without validation

    # ========================================================================
    # EXTRACT GARCH/JUMP PARAMETERS FROM FEATS (v7.0: Unified MC)
    # ========================================================================
    # Previously BMA used run_regime_specific_mc (constant vol, no jumps),
    # while _simulate_forward_paths used GARCH+jumps. This caused p_up and
    # exp_ret to reflect different volatility dynamics — the #1 CRPS killer.
    # Now both come from run_unified_mc with full GARCH+jumps.
    # ========================================================================
    garch_params = feats.get("garch_params", {}) or {}
    _use_garch = isinstance(garch_params, dict) and all(
        k in garch_params for k in ("omega", "alpha", "beta")
    )
    if _use_garch:
        _garch_omega = float(max(garch_params.get("omega", 0.0), 1e-12))
        _garch_alpha = float(np.clip(garch_params.get("alpha", 0.0), 0.0, 0.999))
        _garch_beta = float(np.clip(garch_params.get("beta", 0.0), 0.0, 0.999))
    else:
        _garch_omega, _garch_alpha, _garch_beta = 0.0, 0.0, 0.0

    # Jump-diffusion calibration from historical returns (same as _simulate_forward_paths)
    _jump_intensity, _jump_mean, _jump_std = 0.0, 0.0, 0.05
    _enable_jumps = os.getenv("ENABLE_JUMPS", "true").strip().lower() == "true"
    if _enable_jumps:
        try:
            ret_hist = feats.get("ret", pd.Series(dtype=float))
            vol_hist = feats.get("vol", pd.Series(dtype=float))
            if isinstance(ret_hist, pd.Series) and isinstance(vol_hist, pd.Series) and len(ret_hist) >= 252:
                df_jmp = pd.concat([ret_hist, vol_hist], axis=1, join='inner').dropna()
                if len(df_jmp) >= 252:
                    df_jmp.columns = ['ret', 'vol']
                    z_sc = df_jmp['ret'] / df_jmp['vol']
                    jmask = np.abs(z_sc) > 3.0
                    nj = int(np.sum(jmask))
                    if nj > 0:
                        _jump_intensity = float(nj / len(df_jmp))
                        jr = df_jmp.loc[jmask, 'ret'].values
                        _jump_mean = float(np.mean(jr))
                        _jump_std = float(max(np.std(jr), 0.01))
                    else:
                        _jump_intensity = 0.01
                        _jump_mean = 0.0
                        _jump_std = 0.05
        except Exception:
            _jump_intensity = 0.01
            _jump_mean = 0.0
            _jump_std = 0.05

    # ========================================================================
    # BAYESIAN MODEL AVERAGING: Draw samples from mixture over models
    # ========================================================================
    # p(x | D, r_t) = Σ_m p(x | r_t, m, θ_m) · p(m | r_t)
    #
    # Story 3.5: Importance-weighted sampling replaces the old "append with
    # floor" approach. Model indices are drawn from Categorical(weights),
    # so each path is assigned to exactly one model. This eliminates the
    # MIN_MODEL_SAMPLES=20 floor that caused up to 2.8% representation error.
    #
    # Low-weight models may get 0 samples by chance — that is correct
    # mixture sampling behavior.
    # ========================================================================

    all_samples = []
    all_vol_samples = []
    model_details = {}

    # v7.5: Multi-horizon extraction for calibration fast mode
    if horizons_extract:
        _hz_all_samples = {h: [] for h in horizons_extract}
        _hz_all_vol = {h: [] for h in horizons_extract}

    # Story 3.5: Importance-weighted model index draw
    # First pass: collect valid models and their weights
    _valid_model_names = []
    _valid_model_weights = []
    for _m_name, _m_weight in model_posterior.items():
        _m_params = models.get(_m_name, {})
        if not _m_params.get('fit_success', True):
            continue
        _valid_model_names.append(_m_name)
        _valid_model_weights.append(float(_m_weight))

    # Normalize weights to sum to 1 (defensive)
    _w_arr = np.array(_valid_model_weights, dtype=np.float64)
    _w_sum = _w_arr.sum()
    if _w_sum > 0:
        _w_arr = _w_arr / _w_sum
    elif len(_w_arr) > 0:
        _w_arr = np.ones_like(_w_arr) / len(_w_arr)

    # Draw model indices from Categorical(weights)
    _bma_rng = np.random.default_rng(seed)
    if len(_valid_model_names) > 0:
        _model_indices = _bma_rng.choice(len(_valid_model_names), size=n_paths, p=_w_arr)
        _model_counts = np.bincount(_model_indices, minlength=len(_valid_model_names))
    else:
        _model_counts = np.array([], dtype=np.int64)

    # Build lookup: model_name -> n_paths_for_this_model
    _model_n_paths = {}
    for _idx, _m_name in enumerate(_valid_model_names):
        _model_n_paths[_m_name] = int(_model_counts[_idx])

    _ha_signal_now = 0.0
    try:
        _ha_series = feats.get("ha_drift_signal", pd.Series(dtype=float))
        if isinstance(_ha_series, pd.Series) and not _ha_series.empty:
            _ha_signal_now = float(_ha_series.dropna().iloc[-1])
        elif _ha_series is not None:
            _ha_arr = np.asarray(_ha_series, dtype=np.float64).reshape(-1)
            if _ha_arr.size:
                _ha_signal_now = float(_ha_arr[np.isfinite(_ha_arr)][-1])
    except Exception:
        _ha_signal_now = 0.0
    if not np.isfinite(_ha_signal_now):
        _ha_signal_now = 0.0

    for model_name, model_weight in model_posterior.items():
        model_params = models.get(model_name, {})

        # Skip failed model fits
        if not model_params.get('fit_success', True):
            continue

        # Extract model-specific parameters
        q_m = model_params.get('q', 1e-6)
        phi_m = model_params.get('phi')
        nu_m = model_params.get('nu')
        c_m = model_params.get('c', 1.0)

        # v7.6: Extract per-model GARCH params (fall back to global)
        # Unified models store per-model tuned GARCH; standard models use global
        garch_omega_m = float(model_params.get('garch_omega', _garch_omega))
        garch_alpha_m = float(model_params.get('garch_alpha', _garch_alpha))
        garch_beta_m = float(model_params.get('garch_beta', _garch_beta))
        garch_leverage_m = float(model_params.get('garch_leverage', 0.0))
        _use_garch_m = _use_garch or (
            garch_omega_m > 1e-12 and garch_alpha_m > 1e-6 and garch_beta_m > 1e-6
        )

        # v7.6: Extract enriched params from tuned unified models
        variance_inflation_m = float(model_params.get('variance_inflation', 1.0))
        mu_drift_m = float(model_params.get('mu_drift', 0.0))
        ind_ha_weight_m = float(model_params.get('ind_ha_drift_weight', 0.0))
        ind_ha_mu_drift_m = 0.0
        if model_params.get('indicator_integrated', False) and abs(ind_ha_weight_m) > 1e-12:
            ind_ha_mu_drift_m = float(ind_ha_weight_m * _ha_signal_now * math.sqrt(max(sigma2_step, 1e-12)))
            mu_drift_m += ind_ha_mu_drift_m
        alpha_asym_m = float(model_params.get('alpha_asym', 0.0))
        k_asym_m = float(model_params.get('k_asym', 2.0))
        risk_premium_m = float(model_params.get('risk_premium_sensitivity', 0.0))

        # v7.7: Extract Tier-2 MC params
        kappa_mean_rev_m = float(model_params.get('kappa_mean_rev', 0.0))
        theta_long_var_m = float(model_params.get('theta_long_var', 0.0))
        crps_sigma_shrinkage_m = float(model_params.get('crps_sigma_shrinkage', 1.0))
        ms_sensitivity_m = float(model_params.get('ms_sensitivity', 0.0))
        q_stress_ratio_m = float(model_params.get('q_stress_ratio', 1.0))
        rough_hurst_m = float(model_params.get('rough_hurst', 0.0))

        # v7.7: Extract Tier-3 MC params
        sigma_eta_m = float(model_params.get('sigma_eta', 0.0))
        t_df_asym_m = float(model_params.get('t_df_asym', 0.0))
        regime_switch_prob_m = float(model_params.get('regime_switch_prob', 0.0))
        gamma_vov_m = float(model_params.get('gamma_vov', 0.0))
        vov_damping_m = float(model_params.get('vov_damping', 0.0))
        skew_score_sensitivity_m = float(model_params.get('skew_score_sensitivity', 0.0))
        skew_persistence_m = float(model_params.get('skew_persistence', 0.97))
        loc_bias_var_coeff_m = float(model_params.get('loc_bias_var_coeff', 0.0))
        loc_bias_drift_coeff_m = float(model_params.get('loc_bias_drift_coeff', 0.0))
        q_vol_coupling_m = float(model_params.get('q_vol_coupling', 0.0))

        # v7.8: Elite MC enhancements (dynamic leverage, liquidity stress)
        leverage_dynamic_decay_m = float(model_params.get('leverage_dynamic_decay', 0.0))
        liq_stress_coeff_m = float(model_params.get('liq_stress_coeff', 0.0))

        # v7.8: Per-model Hansen/CST pipeline augmentations (March 2026)
        # Prefer per-model values from internal pipeline; fall back to global
        _hansen_act_m = model_params.get('hansen_activated', False)
        hansen_lambda_m = float(model_params.get('hansen_lambda', 0.0)) if _hansen_act_m else None
        if hansen_lambda_m is not None and abs(hansen_lambda_m) <= 0.01:
            hansen_lambda_m = None
        # Fallback to global for non-pipeline models
        if hansen_lambda_m is None and hansen_skew_t_enabled:
            hansen_lambda_m = hansen_lambda_global

        _cst_act_m = model_params.get('cst_activated', False)
        cst_nu_crisis_m = model_params.get('cst_nu_crisis') if _cst_act_m else None
        cst_epsilon_m = float(model_params.get('cst_epsilon', 0.0)) if _cst_act_m else None
        if cst_nu_crisis_m is not None:
            cst_nu_crisis_m = float(cst_nu_crisis_m)
            cst_nu_normal_m = nu_m  # Use model's own nu as normal component
        else:
            cst_nu_normal_m = None
            cst_epsilon_m = None
        # Fallback to global for non-pipeline models
        if cst_nu_crisis_m is None and cst_enabled:
            cst_nu_normal_m = cst_nu_normal_global
            cst_nu_crisis_m = cst_nu_crisis_global
            cst_epsilon_m = cst_epsilon_global

        # v7.6: Extract per-model jump params (fall back to global)
        jump_intensity_m = float(model_params.get('jump_intensity', _jump_intensity if _enable_jumps else 0.0))
        jump_mean_m = float(model_params.get('jump_mean', _jump_mean))
        # jump_variance → jump_std conversion
        jump_var_m = model_params.get('jump_variance')
        if jump_var_m is not None and jump_var_m > 0:
            jump_std_m = float(math.sqrt(jump_var_m))
        else:
            jump_std_m = float(_jump_std)

        # Default phi for models without it
        if phi_m is None or not np.isfinite(phi_m):
            phi_m = 0.95 if 'phi' in model_name else 1.0

        # ================================================================
        # v7.9: FLOOR phi AT 0 — prevent anti-persistent drift in MC
        # ================================================================
        # Negative phi causes drift to oscillate sign each step, which is
        # statistically possible but produces pathological MC paths:
        # large initial drift × negative phi → huge opposite return next step.
        # Floor at 0 converts anti-persistence to instant mean-reversion,
        # which is the economically sensible interpretation.
        # ================================================================
        if phi_m < 0.0:
            phi_m = 0.0

        # ================================================================
        # v7.9: FLOOR phi AT 0 — prevent anti-persistent drift in MC
        # ================================================================
        # Negative phi causes drift to oscillate sign each step, which is
        # statistically possible but produces pathological MC paths:
        # large initial drift × negative phi → huge opposite return next step.
        # Floor at 0 converts anti-persistence to instant mean-reversion,
        # which is the economically sensible interpretation.
        # ================================================================
        if phi_m < 0.0:
            phi_m = 0.0

        # Validate nu
        if nu_m is not None and (not np.isfinite(nu_m) or nu_m <= 2.0):
            nu_m = None

        # ================================================================
        # GPD-BASED ν ADJUSTMENT — REAL TAIL IMPROVEMENT (March 2026)
        # ================================================================
        # The GPD shape parameter ξ from tuning reveals the TRUE tail
        # heaviness of the data. Student-t tail index: ξ = 1/ν.
        #
        # If GPD shows heavier tails than the model assumes (ξ_gpd > 1/ν),
        # reduce effective ν → MC simulation produces heavier-tailed samples
        # → more conservative risk assessment → better drawdown avoidance.
        #
        # Only adjust downward (more conservative), never upward.
        # ================================================================
        if nu_m is not None:
            _m_cal = model_params.get('calibration_params', {})
            _nu_eff = _m_cal.get('nu_effective')
            if _nu_eff is not None and isinstance(_nu_eff, (int, float)):
                _nu_eff = float(_nu_eff)
                if np.isfinite(_nu_eff) and _nu_eff > 2.5:
                    # Only make tails heavier (lower ν), never lighter
                    if _nu_eff < nu_m:
                        nu_m = _nu_eff

        # Story 3.5: Importance-weighted sample count from categorical draw
        n_model_samples = _model_n_paths.get(model_name, 0)
        if n_model_samples == 0:
            # Model received 0 paths from categorical sampling — skip MC
            model_details[model_name] = {
                "weight": float(model_weight),
                "n_samples": 0,
            }
            continue

        # v7.6: Use run_unified_mc with enriched per-model params
        mc_result = run_unified_mc(
            mu_t=mu_t,
            P_t=P_t,
            phi=phi_m,
            q=q_m,
            sigma2_step=sigma2_step * c_m,
            H_max=H,
            n_paths=n_model_samples,
            nu=nu_m,
            use_garch=_use_garch_m,
            garch_omega=garch_omega_m,
            garch_alpha=garch_alpha_m,
            garch_beta=garch_beta_m,
            jump_intensity=jump_intensity_m,
            jump_mean=jump_mean_m,
            jump_std=jump_std_m,
            # Exotic: Hansen, CST fall back to Python
            hansen_lambda=hansen_lambda_m,
            cst_nu_normal=cst_nu_normal_m if cst_nu_crisis_m is not None else None,
            cst_nu_crisis=cst_nu_crisis_m,
            cst_epsilon=cst_epsilon_m,
            # v7.6: Enriched MC params from tuned models
            garch_leverage=garch_leverage_m,
            variance_inflation=variance_inflation_m,
            mu_drift=mu_drift_m,
            alpha_asym=alpha_asym_m,
            k_asym=k_asym_m,
            risk_premium_sensitivity=risk_premium_m,
            # v7.7: Tier-2 MC params
            kappa_mean_rev=kappa_mean_rev_m,
            theta_long_var=theta_long_var_m,
            crps_sigma_shrinkage=crps_sigma_shrinkage_m,
            ms_sensitivity=ms_sensitivity_m,
            q_stress_ratio=q_stress_ratio_m,
            rough_hurst=rough_hurst_m,
            # v7.7: Tier-3 MC params
            sigma_eta=sigma_eta_m,
            t_df_asym=t_df_asym_m,
            regime_switch_prob=regime_switch_prob_m,
            gamma_vov=gamma_vov_m,
            vov_damping=vov_damping_m,
            skew_score_sensitivity=skew_score_sensitivity_m,
            skew_persistence=skew_persistence_m,
            loc_bias_var_coeff=loc_bias_var_coeff_m,
            loc_bias_drift_coeff=loc_bias_drift_coeff_m,
            q_vol_coupling=q_vol_coupling_m,
            # v7.8: Elite MC enhancements
            leverage_dynamic_decay=leverage_dynamic_decay_m,
            liq_stress_coeff=liq_stress_coeff_m,
            # Story 1.4: Dual-frequency drift
            phi_slow=phi_slow,
            mu_slow_0=mu_slow_0,
            # Story 3.4: Asset-class-aware return cap
            return_cap=return_cap,
        )
        # v7.5: Collect samples at multiple horizons for calibration fast mode
        if horizons_extract:
            for _hz in horizons_extract:
                if _hz <= H:  # H = max(horizons_extract) when called from fast mode
                    _hz_all_samples[_hz].append(mc_result['returns'][_hz - 1, :])
                    _hz_all_vol[_hz].append(mc_result['volatility'][_hz - 1, :])
        # Extract samples at target horizon H (last row)
        model_samples = mc_result['returns'][H - 1, :]
        model_vol_samples = mc_result['volatility'][H - 1, :]

        all_samples.append(model_samples)
        all_vol_samples.append(model_vol_samples)
        model_details[model_name] = {
            "weight": float(model_weight),
            "n_samples": len(model_samples),
            "q": float(q_m),
            "phi": float(phi_m) if phi_m is not None else None,
            "nu": float(nu_m) if nu_m is not None else None,
            "c": float(c_m),
            # v7.6: Enriched per-model params
            "garch_omega": float(garch_omega_m),
            "garch_alpha": float(garch_alpha_m),
            "garch_beta": float(garch_beta_m),
            "garch_leverage": float(garch_leverage_m),
            "variance_inflation": float(variance_inflation_m),
            "mu_drift": float(mu_drift_m),
            "indicator_integrated": bool(model_params.get('indicator_integrated', False)),
            "ind_ha_drift_weight": float(ind_ha_weight_m),
            "ind_ha_mu_drift": float(ind_ha_mu_drift_m),
            "alpha_asym": float(alpha_asym_m),
            "risk_premium": float(risk_premium_m),
            # v7.7: Tier-2 per-model params
            "kappa_mean_rev": float(kappa_mean_rev_m),
            "theta_long_var": float(theta_long_var_m),
            "crps_sigma_shrinkage": float(crps_sigma_shrinkage_m),
            "ms_sensitivity": float(ms_sensitivity_m),
            "q_stress_ratio": float(q_stress_ratio_m),
            "rough_hurst": float(rough_hurst_m),
            # v7.7: Tier-3 per-model params
            "sigma_eta": float(sigma_eta_m),
            "t_df_asym": float(t_df_asym_m),
            "regime_switch_prob": float(regime_switch_prob_m),
            "gamma_vov": float(gamma_vov_m),
            "vov_damping": float(vov_damping_m),
            "skew_score_sensitivity": float(skew_score_sensitivity_m),
            "skew_persistence": float(skew_persistence_m),
            "loc_bias_var_coeff": float(loc_bias_var_coeff_m),
            "loc_bias_drift_coeff": float(loc_bias_drift_coeff_m),
            "q_vol_coupling": float(q_vol_coupling_m),
            # Augmentation layer info
            "hansen_lambda": float(hansen_lambda_global) if hansen_skew_t_enabled else None,
            "cst_enabled": cst_enabled,
        }

    # Concatenate all model samples
    if all_samples:
        r_samples = np.concatenate(all_samples)
        vol_samples = np.concatenate(all_vol_samples)
    else:
        return np.array([0.0]), np.array([0.0]), soft_regime_probs, {
            "method": "FAILED",
            "reason": "no_valid_model_samples",
            "current_regime": current_regime,
            "regime_name": regime_name,
        }

    metadata = {
        "method": "bayesian_model_averaging",
        "has_bma": True,
        "current_regime": current_regime,
        "regime_name": regime_name,
        "is_fallback": is_fallback,
        "model_posterior": {m: float(w) for m, w in model_posterior.items()},
        "model_details": model_details,
        "n_total_samples": len(r_samples),
        # Hansen Skew-t diagnostics
        "hansen_skew_t_enabled": hansen_skew_t_enabled,
        "hansen_lambda": float(hansen_lambda_global) if hansen_lambda_global is not None else None,
        "hansen_nu": float(hansen_nu_global) if hansen_nu_global is not None else None,
        "hansen_skew_direction": (
            "left" if hansen_lambda_global and hansen_lambda_global < -0.01 else
            ("right" if hansen_lambda_global and hansen_lambda_global > 0.01 else "symmetric")
        ) if hansen_lambda_global is not None else "not_available",
        # Epistemic weighting diagnostics
        "posteriors_recomputed": posteriors_recomputed,
        "cached_posterior": {m: float(w) for m, w in cached_posterior.items()} if posteriors_recomputed else None,
        "epistemic_weighting": epistemic_meta,
        # Model selection diagnostics
        "model_selection_method": regime_meta.get('model_selection_method', 'combined'),
        "effective_selection_method": epistemic_meta.get('method', 'combined'),
        "hyvarinen_disabled": regime_meta.get('hyvarinen_disabled', False),
        "bic_weight": regime_meta.get('bic_weight', 0.5),
        "entropy_lambda": regime_meta.get('entropy_lambda', 0.05),
        "hyvarinen_max": regime_meta.get('hyvarinen_max'),
        "combined_score_min": regime_meta.get('combined_score_min'),
        "bic_min": regime_meta.get('bic_min'),
        # Contaminated Student-t diagnostics
        "contaminated_student_t_enabled": cst_enabled,
        "cst_nu_normal": float(cst_nu_normal_global) if cst_nu_normal_global is not None else None,
        "cst_nu_crisis": float(cst_nu_crisis_global) if cst_nu_crisis_global is not None else None,
        "cst_epsilon": float(cst_epsilon_global) if cst_epsilon_global is not None else None,
        # SOFT REGIME PROBABILITIES FOR TRUST AUTHORITY
        # Used by CalibratedTrust to avoid penalty cliffs at regime boundaries
        "soft_regime_probs": soft_regime_probs,
    }

    # v7.5: Add per-horizon samples to metadata for multi-horizon calibration
    if horizons_extract:
        metadata["horizon_samples"] = {
            h: np.concatenate(_hz_all_samples[h]) if _hz_all_samples[h] else np.array([0.0])
            for h in horizons_extract
        }
        metadata["horizon_vol_samples"] = {
            h: np.concatenate(_hz_all_vol[h]) if _hz_all_vol[h] else np.array([0.0])
            for h in horizons_extract
        }

    # Return soft regime probs dict for trust authority (not legacy array)
    return r_samples, vol_samples, soft_regime_probs, metadata
