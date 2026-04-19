"""
Asset processing module: process_single_asset worker and helpers.

Extracted from signals.py (Story 8.6).
"""
from __future__ import annotations

import os
import sys

# Ensure src directory is in path for imports
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Feature flags, conditional imports, and utility functions from config
# ---------------------------------------------------------------------------
from decision.signal_modules.config import *  # noqa: F401,F403
from decision.signal_modules.config import (  # Private names
    _to_float, _download_prices,
)

# ---------------------------------------------------------------------------
# Sibling module imports for functions called by process_single_asset
# ---------------------------------------------------------------------------
from decision.signal_modules.data_fetching import *  # noqa: F401,F403
from decision.signal_modules.data_fetching import (  # Private names
    _garch11_mle, _fit_student_nu_mle,
)
from decision.signal_modules.regime_classification import *  # noqa: F401,F403
from decision.signal_modules.parameter_loading import *  # noqa: F403
from decision.signal_modules.parameter_loading import (  # Private names
    _safe_get_nested, _load_tuned_kalman_params, _select_regime_params,
)
from decision.signal_modules.feature_pipeline import *  # noqa: F403
from decision.signal_modules.signal_generation import *  # noqa: F403
from decision.signal_modules.comprehensive_diagnostics import compute_all_diagnostics  # noqa: E402


def _enrich_signal_with_epic8(
    asset: str,
    sigs: list,
    tuned_params: Optional[Dict],
    feats: Optional[Dict] = None,
) -> Dict:
    """
    Post-processing enrichment from Epic 8 modules.

    Adds conviction score, Kelly sizing, and signal TTL to the result dict.
    All enrichments are optional and degrade gracefully.

    Args:
        asset: Symbol.
        sigs: List of Signal objects from latest_signals().
        tuned_params: Tuned params dict (contains BMA weights).
        feats: Feature dict from compute_features().

    Returns:
        Dict with enrichment data (may be empty if modules unavailable).
    """
    enrichment: Dict = {}

    if not sigs:
        return enrichment

    # --- 8.3: Conviction Scoring ---
    if CONVICTION_SCORING_AVAILABLE and tuned_params:
        try:
            # Extract BMA weights from tuned params
            bma_weights = {}
            global_data = tuned_params.get("global") or {}
            bma_weights = global_data.get("model_posterior", {})

            # Get regime fit from first signal
            regime_fit = 1.0 - sigs[0].pit_violation_severity if sigs else 0.5

            # Forecast stability from multiple horizon forecasts
            forecasts_by_horizon = {}
            for s in sigs:
                forecasts_by_horizon[s.horizon_days] = [s.exp_ret]

            conviction = compute_conviction(
                symbol=asset,
                bma_weights=bma_weights if bma_weights else {"default": 1.0},
                regime_fit=float(regime_fit),
                recent_hits=[],  # Historical accuracy requires past forecasts
                forecasts_by_horizon=forecasts_by_horizon,
            )
            enrichment["conviction"] = {
                "composite": conviction.composite,
                "category": conviction.category,
                "model_agreement": conviction.model_agreement,
                "regime_confidence": conviction.regime_confidence,
                "forecast_stability": conviction.forecast_stability,
            }
        except Exception:
            pass

    # --- 8.7: Kelly Sizing ---
    if KELLY_SIZING_AVAILABLE:
        try:
            kelly_recs = []
            for s in sigs:
                if s.p_up > 0.01 and s.p_up < 0.99 and s.expected_gain > 0:
                    rec = recommend_position_size(
                        symbol=asset,
                        p_win=float(s.p_up),
                        avg_win=float(s.expected_gain),
                        avg_loss=float(max(s.expected_loss, 1e-6)),
                    )
                    kelly_recs.append({
                        "horizon": s.horizon_days,
                        "half_kelly": rec.half_kelly,
                        "capped_size": rec.capped_size,
                        "edge": rec.edge,
                    })
            if kelly_recs:
                enrichment["kelly"] = kelly_recs
        except Exception:
            pass

    # --- 8.8: Signal Decay / TTL ---
    if SIGNAL_DECAY_AVAILABLE:
        try:
            decay_info = []
            for s in sigs:
                hl = compute_half_life(s.horizon_days)
                decay_info.append({
                    "horizon": s.horizon_days,
                    "half_life_days": hl,
                    "ttl_at_generation": hl * 3.32,  # log2(10) * hl ≈ time to 10% strength
                })
            enrichment["signal_ttl"] = decay_info
        except Exception:
            pass

    # --- Epic 27: Forecast Attribution ---
    if FORECAST_ATTRIBUTION_AVAILABLE and feats is not None:
        try:
            returns = feats.get("returns") if isinstance(feats, dict) else getattr(feats, "returns", None)
            if returns is not None and len(sigs) > 0:
                mu_forecast = np.array([s.exp_ret for s in sigs])
                sigma_forecast = np.array([max(s.ci_width, 1e-8) for s in sigs if hasattr(s, 'ci_width')])
                if len(mu_forecast) > 0 and len(sigma_forecast) > 0:
                    da = drift_attribution(np.asarray(returns)[-len(mu_forecast):], mu_forecast, sigma_forecast[:len(mu_forecast)])
                    enrichment["forecast_attribution"] = da.to_dict() if hasattr(da, 'to_dict') else {"drift_contribution": float(getattr(da, 'drift_contribution', 0.0))}
        except Exception:
            pass

    return enrichment


# =============================================================================
# ISOTONIC RECALIBRATION HELPER
# =============================================================================
# This function loads a persisted transport map and applies it to PIT values.
# The transport map is learned during tuning and stored in the cache.
#
# DOCTRINE:
#   - Calibration is a FIRST-CLASS PROBABILISTIC TRANSPORT OPERATOR
#   - Applied BEFORE regimes see PIT values
#   - Regimes see CALIBRATED probability, not raw belief
#   - This is NOT a patch/validator/escalation trigger
# =============================================================================

def load_and_apply_recalibration(
    tuned_params: Optional[Dict],
    raw_pit: np.ndarray,
) -> Tuple[np.ndarray, bool, Optional[Dict]]:
    """
    Load recalibration transport map from tuned params and apply to raw PIT.
    
    Args:
        tuned_params: Full tuned params dict containing recalibration info
        raw_pit: Array of raw PIT values from model
        
    Returns:
        Tuple of:
        - calibrated_pit: Array of calibrated PIT values
        - was_recalibrated: True if recalibration was applied
        - recal_meta: Metadata about recalibration (or None)
    """
    if not ISOTONIC_RECALIBRATION_AVAILABLE:
        return raw_pit, False, None
    
    if tuned_params is None:
        return raw_pit, False, None
    
    recal_data = tuned_params.get('recalibration')
    if recal_data is None or not tuned_params.get('recalibration_applied', False):
        return raw_pit, False, None
    
    try:
        # Load transport map from persisted data
        transport_map = TransportMapResult.from_dict(recal_data)
        
        # Check if it's an identity map (no change needed)
        if transport_map.is_identity or transport_map.fallback_to_identity:
            return raw_pit, False, {
                'is_identity': True,
                'reason': 'identity_map' if transport_map.is_identity else 'fallback',
            }
        
        # Apply recalibration
        calibrated_pit = apply_recalibration(raw_pit, transport_map)
        
        recal_meta = {
            'applied': True,
            'n_segments': transport_map.n_segments,
            'ks_improvement': transport_map.ks_improvement,
            'raw_ks_pvalue': transport_map.raw_ks_pvalue,
            'calibrated_ks_pvalue': transport_map.calibrated_ks_pvalue,
        }
        
        return calibrated_pit, True, recal_meta
        
    except Exception as e:
        # If transport map recalibration fails, try pit_recalibration fallback (Epic 26)
        if PIT_RECALIBRATION_AVAILABLE and len(raw_pit) >= 50:
            try:
                recal_result = isotonic_recalibrate(raw_pit)
                if recal_result.recalibrated_pit is not None:
                    return np.asarray(recal_result.recalibrated_pit), True, {
                        'applied': True,
                        'method': 'pit_recalibration_fallback',
                        'transport_map_error': str(e),
                    }
            except Exception:
                pass
        if os.getenv('DEBUG'):
            print(f"Warning: Recalibration failed: {e}")
        return raw_pit, False, {'error': str(e)}


def process_single_asset(args_tuple: Tuple) -> Optional[Dict]:
    """
    Worker function to process a single asset in parallel.
    Only performs computation, no console output.

    Args:
        args_tuple: (asset, args, horizons)

    Returns:
        Dictionary with processed results or None if failed
    """
    asset, args, horizons = args_tuple

    try:
        # Fetch price data
        try:
            px, title = fetch_px_asset(asset, args.start, args.end)
        except Exception as e:
            return {
                "status": "error",
                "asset": asset,
                "error": str(e)
            }

        # De-duplicate by resolved symbol
        canon = extract_symbol_from_title(title)
        if not canon:
            canon = asset.strip().upper()

        # Fetch OHLC data for Garman-Klass volatility (7.4x more efficient)
        ohlc_df = None
        if GK_VOLATILITY_AVAILABLE:
            try:
                ohlc_df = _download_prices(asset, args.start, args.end)
            except Exception:
                ohlc_df = None  # Fall back to GARCH/EWMA

        # Compute features and signals with OHLC for GK volatility
        feats = compute_features(px, asset_symbol=asset, ohlc_df=ohlc_df)
        last_close = _to_float(px.iloc[-1])

        # Data quality assessment (Epic 29: missing_data)
        data_quality_meta = None
        if MISSING_DATA_AVAILABLE:
            try:
                returns = feats.get("returns") if isinstance(feats, dict) else getattr(feats, "returns", None)
                if returns is not None:
                    dq_result = data_quality_score(np.asarray(returns), expected_obs=252)
                    data_quality_meta = dq_result.to_dict() if hasattr(dq_result, 'to_dict') else {"score": float(getattr(dq_result, 'score', 1.0))}
            except Exception:
                pass

        # Load tuned params with BMA structure for model averaging
        tuned_params = _load_tuned_kalman_params(asset)

        # Detect asset type for CI bounding (March 2026)
        _asset_type = classify_asset_type(asset)

        # Pass asset_key (canon) for display price inertia (Upgrade #3)
        sigs, thresholds = latest_signals(feats, horizons, last_close=last_close, t_map=args.t_map, ci=args.ci, tuned_params=tuned_params, asset_key=canon, asset_type=_asset_type)

        # Compute diagnostics if requested
        diagnostics = {}
        if args.diagnostics or args.diagnostics_lite or args.pit_calibration or args.model_comparison:
            enable_oos = args.diagnostics
            enable_pit = args.pit_calibration
            enable_model_comp = args.model_comparison
            diagnostics = compute_all_diagnostics(px, feats, enable_oos=enable_oos, enable_pit_calibration=enable_pit, enable_model_comparison=enable_model_comp)

        # Epic 8 enrichment: conviction scoring, Kelly sizing, signal TTL
        enrichment = _enrich_signal_with_epic8(asset, sigs, tuned_params, feats)

        # =============================================================
        # Signal-side story integrations (enrichment metadata)
        # =============================================================
        _signal_meta = {}
        returns_arr = feats.get("returns")
        vol_arr = feats.get("vol")
        _has_returns = returns_arr is not None and len(returns_arr) > 0
        _has_vol = vol_arr is not None and len(vol_arr) > 0

        # Story 10.1-10.3: Sign probability with uncertainty
        if SIGN_PROBABILITY_AVAILABLE and _has_returns and _has_vol and tuned_params:
            try:
                _tp_global = tuned_params.get("global", {})
                _sp_mu = _tp_global.get("mu", 0.0)
                _sp_P = _tp_global.get("P", 1e-4)
                _sp_c = _tp_global.get("c", 1.0)
                _sp_sigma = float(vol_arr.iloc[-1])
                _sp_nu = _tp_global.get("nu")
                # sign_prob_with_uncertainty returns a float
                _sp_val = sign_prob_with_uncertainty(
                    _sp_mu, _sp_P, _sp_sigma, c=_sp_c,
                    model='student_t' if _sp_nu else 'gaussian',
                    nu=_sp_nu,
                )
                _signal_meta["sign_prob"] = float(_sp_val)
                if _sp_nu is not None:
                    # sign_prob_skewed: (mu_t, P_t, sigma_t, nu_L, nu_R, c)
                    _sp_skew_val = sign_prob_skewed(_sp_mu, _sp_P, _sp_sigma, nu_L=float(_sp_nu), nu_R=float(_sp_nu), c=_sp_c)
                    _signal_meta["sign_prob_skewed"] = float(_sp_skew_val)
                # Story 5.3: Multi-horizon sign probabilities
                _sp_phi = float(_tp_global.get("phi", 0.0))
                _mh_model = 'student_t' if _sp_nu else 'gaussian'
                _mh_probs = {}
                for _h in (1, 3, 7, 30):
                    _mh_probs[f"H{_h}"] = float(multi_horizon_sign_prob(
                        _sp_mu, _sp_P, _sp_phi, _sp_sigma, _sp_c, _h,
                        model=_mh_model, nu=_sp_nu,
                    ))
                _signal_meta["multi_horizon_sign_prob"] = _mh_probs
            except Exception:
                pass

        # Story 11.1: Laplace posterior approximation
        if LAPLACE_POSTERIOR_AVAILABLE and tuned_params:
            try:
                _tp_global = tuned_params.get("global", {})
                _lp_nu = _tp_global.get("nu")
                _lp = laplace_posterior(
                    np.asarray(returns_arr) if _has_returns else np.array([0.0]),
                    np.asarray(vol_arr) if _has_vol else np.array([0.01]),
                    np.array([_tp_global.get("c", 1.0), _tp_global.get("phi", 0.0), _tp_global.get("q", 1e-5)]),
                    family="student_t" if _lp_nu else "gaussian",
                    nu=float(_lp_nu) if _lp_nu else None,
                )
                _signal_meta["laplace_posterior"] = {
                    "mu_mode": float(_lp.mu_mode),
                    "sigma_mode": float(_lp.sigma_mode),
                }
            except Exception:
                pass

        # Story 12.1-12.2: MC variance reduction
        if MC_VARIANCE_REDUCTION_AVAILABLE and tuned_params:
            try:
                _tp_global = tuned_params.get("global", {})
                _mc_mu = _tp_global.get("mu", 0.0)
                _mc_vol = float(vol_arr.iloc[-1]) if _has_vol else 0.01
                _mc_nu = _tp_global.get("nu", 8.0)  # Default nu for antithetic
                if _mc_nu is not None:
                    _mc_is = importance_mc_student_t(
                        _mc_mu, _mc_vol, float(_mc_nu), n_samples=5000,
                    )
                    _signal_meta["mc_importance_sampling"] = {
                        "ess": float(_mc_is.ess),
                        "ess_ratio": float(_mc_is.ess_ratio),
                    }
                    _mc_at = antithetic_mc_sample(_mc_mu, _mc_vol, float(_mc_nu), n_samples=5000)
                    _signal_meta["mc_antithetic"] = {
                        "var_reduction": float(_mc_at.var_reduction),
                    }
            except Exception:
                pass

        # Story 13.1: Platt calibration
        if PLATT_CALIBRATE_AVAILABLE and sigs:
            try:
                _raw_probs = [s.p_up for s in sigs]
                _outcomes = [1.0 if s.exp_ret > 0 else 0.0 for s in sigs]
                if len(_raw_probs) >= 2:
                    _pc = platt_calibrate(np.array(_raw_probs), np.array(_outcomes))
                    _signal_meta["platt_calibration"] = {
                        "a": float(_pc.a),
                        "b": float(_pc.b),
                    }
            except Exception:
                pass

        # Story 14.1: Uncertainty decomposition
        if UNCERTAINTY_DECOMPOSITION_AVAILABLE and tuned_params:
            try:
                _tp_global = tuned_params.get("global", {})
                _models_info = tuned_params.get("models", {})
                if _models_info:
                    _model_means = np.array([_models_info[m].get("mu", 0.0) for m in _models_info if _models_info[m].get("fit_success", False)])
                    _model_stds = np.array([np.sqrt(_models_info[m].get("P", 1e-4)) for m in _models_info if _models_info[m].get("fit_success", False)])
                    _model_wts = np.array([tuned_params.get("model_posterior", {}).get(m, 0.0) for m in _models_info if _models_info[m].get("fit_success", False)])
                    if len(_model_means) >= 2 and _model_wts.sum() > 0:
                        _model_wts = _model_wts / _model_wts.sum()
                        _ud = decompose_uncertainty(_model_means, _model_stds, _model_wts)
                        _signal_meta["uncertainty_decomposition"] = {
                            "epistemic": float(_ud.epistemic_var) if hasattr(_ud, 'epistemic_var') else 0.0,
                            "aleatoric": float(_ud.aleatoric_var) if hasattr(_ud, 'aleatoric_var') else 0.0,
                            "total": float(_ud.total_var) if hasattr(_ud, 'total_var') else 0.0,
                            "epistemic_fraction": float(_ud.epistemic_fraction) if hasattr(_ud, 'epistemic_fraction') else 0.0,
                        }
            except Exception:
                pass

        # Story 15.1: Regime confidence scaling (Story 11.3)
        if REGIME_CONFIDENCE_AVAILABLE and sigs and tuned_params:
            try:
                _s0 = sigs[0]
                _rc_confidence = float(_s0.p_up)
                _rc_regime = str(_s0.regime)
                # Use calibration diagnostics for historical hit rates if available
                _rc_diag = (tuned_params.get("global", {}) or {}).get("calibration_params", {})
                _rc_hist_hr_raw = _rc_diag.get("regime_hit_rates")
                if _rc_hist_hr_raw is not None and isinstance(_rc_hist_hr_raw, dict):
                    # Convert raw dict to RegimeHitRates dataclass
                    _rc_hit_rates = RegimeHitRates(
                        rates=_rc_hist_hr_raw.get("rates", {}),
                        counts=_rc_hist_hr_raw.get("counts", {}),
                        global_rate=float(_rc_hist_hr_raw.get("global_rate", 0.5)),
                        total_count=int(_rc_hist_hr_raw.get("total_count", 0)),
                    )
                    _rc_result = regime_confidence_scale(
                        confidence=_rc_confidence,
                        regime=_rc_regime,
                        historical_hit_rates=_rc_hit_rates,
                    )
                    if _rc_result is not None:
                        _signal_meta["regime_confidence"] = {
                            "adjusted_confidence": float(_rc_result.adjusted_confidence),
                            "scale_factor": float(_rc_result.scale_factor),
                            "regime": str(_rc_result.regime),
                        }
            except Exception:
                pass

        # Story 12.1: Adaptive momentum horizon weights
        if MULTI_TIMEFRAME_FUSION_AVAILABLE and _has_returns:
            try:
                _amw = adaptive_momentum_weights(np.asarray(returns_arr))
                _signal_meta["adaptive_momentum_weights"] = {
                    "short_weight": float(_amw.weights[0]),
                    "long_weight": float(_amw.weights[-1]),
                    "best_horizon": int(_amw.best_horizon),
                }
            except Exception:
                pass

        # Story 12.2: Momentum vs mean-reversion regime indicator
        if MULTI_TIMEFRAME_FUSION_AVAILABLE and _has_returns:
            try:
                _vol_np = np.asarray(vol_arr) if _has_vol else None
                _mmr = momentum_mr_regime_indicator(np.asarray(returns_arr), vol=_vol_np)
                _signal_meta["momentum_mr_regime"] = {
                    "regime": str(_mmr.regime),
                    "variance_ratio": float(_mmr.variance_ratio),
                    "is_momentum": bool(_mmr.is_momentum),
                    "is_mean_reverting": bool(_mmr.is_mean_reverting),
                }
            except Exception:
                pass

        # Story 13.1-13.2: Kelly sizing calibration
        if KELLY_SIZING_CALIBRATION_AVAILABLE and sigs and tuned_params:
            try:
                _s0 = sigs[0]
                _tp_global = tuned_params.get("global", {})
                # Story 13.1: Kelly fraction from BMA predictive distribution
                _kf_mu = _tp_global.get("mu", 0.0)
                _kf_sigma = float(vol_arr.iloc[-1]) if _has_vol else 0.02
                _kf_nu = _tp_global.get("nu")
                _kf = kelly_fraction(_kf_mu, _kf_sigma, nu=_kf_nu)
                _signal_meta["kelly_fraction"] = {
                    "f": float(_kf),
                    "mu": float(_kf_mu),
                    "sigma": float(_kf_sigma),
                    "nu": float(_kf_nu) if _kf_nu is not None else None,
                }
                # Story 13.2: drawdown_adjusted_kelly(f_kelly, current_dd, max_dd)
                _dak = drawdown_adjusted_kelly(
                    _kf,
                    current_dd=0.0,  # No current drawdown available at signal time
                    max_dd=0.10,
                )
                _signal_meta["drawdown_adjusted_kelly"] = {
                    "f_adjusted": float(_dak.f_adjusted),
                    "dd_dampener": float(_dak.dd_dampener),
                    "is_flat": bool(_dak.is_flat),
                }
            except Exception:
                pass

        # Story 14.1: Transaction costs
        if TRANSACTION_COSTS_AVAILABLE and sigs:
            try:
                _last_price = float(px.iloc[-1]) if px is not None and len(px) > 0 else 100.0
                _last_vol = float(vol_arr.iloc[-1]) if _has_vol else 0.02
                _tc = transaction_cost(
                    price=_last_price,
                    shares=abs(sigs[0].position_strength) * 100,
                    daily_vol=_last_vol,
                )
                _signal_meta["transaction_cost"] = {
                    "cost_bps": float(_tc.cost_bps),
                    "total_cost": float(_tc.total_cost),
                    "spread_cost": float(_tc.spread_cost),
                    "impact_cost": float(_tc.impact_cost),
                }
            except Exception:
                pass

        # Story 15.1: Regime position limits
        if REGIME_POSITION_SIZING_AVAILABLE and sigs:
            try:
                _sig_regime = sigs[0].regime if sigs[0].regime else "LOW_VOL_TREND"
                _rpl = regime_position_limit(
                    regime=str(_sig_regime),
                    raw_fraction=abs(sigs[0].position_strength),
                )
                _signal_meta["regime_position_limit"] = {
                    "max_fraction": float(_rpl.max_fraction),
                    "was_limited": bool(_rpl.was_limited),
                    "limited_fraction": float(_rpl.limited_fraction),
                }
            except Exception:
                pass

        # Story 15.2: Dynamic leverage from forecast confidence
        if REGIME_POSITION_SIZING_AVAILABLE and sigs:
            try:
                _dl_conf = float(sigs[0].p_up)
                _dl = dynamic_leverage(confidence=_dl_conf)
                _signal_meta["dynamic_leverage"] = float(_dl)
            except Exception:
                pass

        # Story 15.3: Volatility targeting overlay
        if REGIME_POSITION_SIZING_AVAILABLE and _has_vol:
            try:
                _vt_sigma = float(vol_arr.iloc[-1]) * np.sqrt(252)  # annualize
                _vt = vol_target_weight(sigma_asset=_vt_sigma)
                _signal_meta["vol_target"] = {
                    "weight": float(_vt.weight),
                    "sigma_asset": float(_vt.sigma_asset),
                    "was_capped": bool(_vt.was_capped),
                    "was_floored": bool(_vt.was_floored),
                }
            except Exception:
                pass

        # Story 16.3: GARCH variance forecast
        # garch_variance_forecast(omega, alpha, gamma, beta, h_T, horizon)
        # Uses GARCH parameters from tuned_params to forecast variance at multiple horizons.
        if GARCH_FORECAST_AVAILABLE and tuned_params:
            try:
                _tp_global = tuned_params.get("global", {})
                _garch_omega = _tp_global.get("garch_omega")
                _garch_alpha = _tp_global.get("garch_alpha")
                _garch_beta = _tp_global.get("garch_beta")
                _garch_gamma = _tp_global.get("garch_leverage", 0.0)
                if all(v is not None for v in [_garch_omega, _garch_alpha, _garch_beta]):
                    # Use current vol as h_T (most recent variance)
                    _h_T = float(vol_arr.iloc[-1]) ** 2 if _has_vol else 0.0001
                    _gf_1d = garch_variance_forecast(
                        omega=float(_garch_omega), alpha=float(_garch_alpha),
                        gamma=float(_garch_gamma), beta=float(_garch_beta),
                        h_T=_h_T, horizon=1,
                    )
                    _gf_5d = garch_variance_forecast(
                        omega=float(_garch_omega), alpha=float(_garch_alpha),
                        gamma=float(_garch_gamma), beta=float(_garch_beta),
                        h_T=_h_T, horizon=5,
                    )
                    _signal_meta["garch_forecast"] = {
                        "vol_1d": float(np.sqrt(_gf_1d.forecast_var)),
                        "vol_5d": float(np.sqrt(_gf_5d.forecast_var)),
                        "unconditional_vol": float(np.sqrt(_gf_1d.unconditional_var)),
                    }
            except Exception:
                pass

        # Story 23.2: Skew-adjusted direction
        if SKEW_ADJUSTED_DIRECTION_AVAILABLE and tuned_params:
            try:
                _tp_global = tuned_params.get("global", {})
                _sad_mu = _tp_global.get("mu", 0.0)
                _sad_sigma = float(vol_arr.iloc[-1]) if _has_vol else 0.01
                _sad_nu = _tp_global.get("nu", 10.0)
                _sad_lambda = _tp_global.get("hansen_lambda", 0.0)
                # skew_adjusted_direction(mu, sigma, nu, lambda_) -> SkewAdjustedDirectionResult
                _sad_val = skew_adjusted_direction(_sad_mu, _sad_sigma, float(_sad_nu), _sad_lambda)
                _signal_meta["skew_adjusted_direction"] = {
                    "prob_positive": float(_sad_val.prob_positive),
                    "prob_positive_symmetric": float(_sad_val.prob_positive_symmetric),
                    "skew_adjustment": float(_sad_val.skew_adjustment),
                }
            except Exception:
                pass

        # Story 18.2-18.3: CST jump probability & prediction interval
        # cst_jump_probability(r_t, mu_t, sigma_t, epsilon, nu_normal, nu_crisis) -> JumpProbabilityResult
        # cst_prediction_interval(mu, sigma, epsilon, nu_normal, nu_crisis, alpha) -> CSTPredictionInterval
        # Uses latest observation and CST parameters from tuned_params.
        if CST_SIGNALS_AVAILABLE and _has_returns and _has_vol and tuned_params:
            try:
                _tp_global = tuned_params.get("global", {})
                _cst_epsilon = _tp_global.get("cst_epsilon")
                _cst_nu_normal = _tp_global.get("cst_nu_normal", _tp_global.get("nu"))
                _cst_nu_crisis = _tp_global.get("cst_nu_crisis")
                _cst_mu = _tp_global.get("mu", 0.0)
                _cst_sigma = float(vol_arr.iloc[-1]) if _has_vol else 0.01
                _cst_r_t = float(returns_arr.iloc[-1]) if hasattr(returns_arr, 'iloc') else float(returns_arr[-1])
                if _cst_epsilon is not None and _cst_nu_normal is not None and _cst_nu_crisis is not None:
                    _cjp = cst_jump_probability(
                        r_t=_cst_r_t, mu_t=float(_cst_mu), sigma_t=_cst_sigma,
                        epsilon=float(_cst_epsilon),
                        nu_normal=float(_cst_nu_normal), nu_crisis=float(_cst_nu_crisis),
                    )
                    _signal_meta["cst_jump_probability"] = {
                        "gamma": float(_cjp.gamma),
                        "is_jump": bool(_cjp.is_jump),
                        "q_inflation": float(_cjp.q_inflation),
                    }
            except Exception:
                pass
            try:
                if _cst_epsilon is not None and _cst_nu_normal is not None and _cst_nu_crisis is not None:
                    _cpi = cst_prediction_interval(
                        mu=float(_cst_mu), sigma=_cst_sigma,
                        epsilon=float(_cst_epsilon),
                        nu_normal=float(_cst_nu_normal), nu_crisis=float(_cst_nu_crisis),
                    )
                    _signal_meta["cst_prediction_interval"] = {
                        "q_lo": float(_cpi.q_lo),
                        "q_hi": float(_cpi.q_hi),
                        "width": float(_cpi.width),
                        "width_ratio": float(_cpi.width_ratio),
                    }
            except Exception:
                pass

        # Story 25.1: Mean reversion signal strength
        if MR_SIGNAL_STRENGTH_AVAILABLE and _has_returns:
            try:
                _tp_global = tuned_params.get("global", {}) if tuned_params else {}
                _mr_price = float(px.iloc[-1]) if px is not None and len(px) > 0 else 100.0
                _mr_eq = _tp_global.get("equilibrium", _mr_price)
                _mr_kappa = _tp_global.get("kappa", 0.1)
                _mr_sigma = float(vol_arr.iloc[-1]) if _has_vol else 0.02
                _mrs = mr_signal_strength(_mr_price, _mr_eq, _mr_kappa, _mr_sigma)
                _signal_meta["mr_signal_strength"] = {
                    "z_score": float(_mrs.z),
                    "strength": str(_mrs.strength),
                    "kelly_fraction": float(_mrs.kelly_fraction),
                    "direction": int(_mrs.direction),
                    "distance": float(_mrs.distance),
                }
            except Exception:
                pass

        # Story 26.1-26.2: Factor augmented signals
        if FACTOR_AUGMENTED_AVAILABLE and _has_returns:
            try:
                # extract_market_factors needs cross-sectional data (2D, N>=5 assets)
                # Single-asset PCA is mathematically impossible; skip gracefully
                _ret_2d = np.asarray(returns_arr).reshape(-1, 1)
                if _ret_2d.shape[1] < 5:
                    raise ValueError("PCA requires >= 5 assets; single-asset not supported")
                _emf = extract_market_factors(_ret_2d, n_factors=1)
                _signal_meta["market_factors"] = {
                    "n_factors": int(_emf.n_factors),
                    "explained_variance_ratio": float(_emf.cumulative_variance),
                }
            except Exception:
                pass

        # Story 23.1-23.3: VIX forecast adjustment
        # vix_drift_adjustment(mu_t, vix_current, vix_median) -> VIXDriftAdjustmentResult
        # vix_term_structure_vol(vix_30, vix_90, horizon) -> VIXTermStructureResult
        # Fetches current VIX from market_conditioning module.
        if VIX_FORECAST_ADJUSTMENT_AVAILABLE and tuned_params:
            try:
                _vix_current = None
                try:
                    from calibration.market_conditioning import get_current_vix
                    _vix_current = get_current_vix()
                except Exception:
                    pass
                if _vix_current is not None:
                    _tp_global = tuned_params.get("global", {})
                    _vda_mu = _tp_global.get("mu", 0.0)
                    _vda = vix_drift_adjustment(
                        mu_t=float(_vda_mu), vix_current=float(_vix_current),
                    )
                    _signal_meta["vix_drift_adjustment"] = {
                        "dampening_factor": float(_vda.dampening_factor),
                        "adjustment_applied": bool(_vda.adjustment_applied),
                        "regime": _vda.regime,
                        "vix_current": float(_vix_current),
                    }
                    # Also compute term structure vol at key horizons
                    try:
                        _vts_7d = vix_term_structure_vol(
                            vix_30=float(_vix_current), vix_90=float(_vix_current) * 0.95,
                            horizon=7,
                        )
                        _signal_meta["vix_term_structure_7d"] = {
                            "implied_vol": float(_vts_7d.implied_vol),
                            "term_structure_state": _vts_7d.term_structure_state,
                        }
                    except Exception:
                        pass
            except Exception:
                pass

        # Story 28.1-28.3: Ensemble forecasts
        if ENSEMBLE_FORECAST_AVAILABLE and sigs:
            try:
                _forecasts = np.array([s.exp_ret for s in sigs])
                _ewe = equal_weight_ensemble(_forecasts)
                _signal_meta["ensemble_equal_weight"] = {
                    "forecast": float(_ewe.forecast),
                    "variance": float(_ewe.variance),
                    "model_spread": float(_ewe.model_spread),
                    "n_models": int(_ewe.n_models),
                }
                if len(_forecasts) >= 3:
                    _te = trimmed_ensemble(_forecasts)
                    _signal_meta["ensemble_trimmed"] = {
                        "forecast": float(_te.forecast),
                        "variance": float(_te.variance),
                        "n_trimmed": int(_te.n_trimmed),
                        "model_spread_trimmed": float(_te.model_spread_trimmed),
                    }
            except Exception:
                pass

        # Story 29.1: Location-scale correction
        if LOCATION_SCALE_CORRECTION_AVAILABLE and _has_returns and _has_vol and tuned_params:
            try:
                _tp_global = tuned_params.get("global", {})
                _lsc_c = _tp_global.get("c", 1.0)
                _lsc_q = _tp_global.get("q", 1e-5)
                _lsc_phi = _tp_global.get("phi", 0.0)
                # Run filter to get innovations
                from models.gaussian import kalman_filter_drift_phi
                _lsc_mu, _lsc_P, _ = kalman_filter_drift_phi(
                    np.asarray(returns_arr), np.asarray(vol_arr), _lsc_q, _lsc_c, _lsc_phi,
                )
                _lsc_innovations = np.asarray(returns_arr) - _lsc_mu
                _lsc_R = _lsc_c * np.asarray(vol_arr) ** 2
                _lsc = location_scale_correction(_lsc_innovations, _lsc_R)
                _signal_meta["location_scale_correction"] = {
                    "delta_mu": float(_lsc.delta_mu) if hasattr(_lsc, 'delta_mu') else 0.0,
                    "scale_sigma": float(_lsc.scale_sigma) if hasattr(_lsc, 'scale_sigma') else 1.0,
                }
            except Exception:
                pass

        # Story 30.1: BMA attribution
        if BMA_ATTRIBUTION_AVAILABLE and tuned_params and _has_returns:
            try:
                _bma_models = tuned_params.get("models", {})
                _bma_weights_arr = tuned_params.get("model_posterior", {})
                if _bma_models and _bma_weights_arr:
                    _bma_model_names = [m for m in _bma_models if _bma_models[m].get("fit_success", False)]
                    if len(_bma_model_names) >= 2:
                        # bma_attribution(returns, model_forecasts, bma_weights)
                        # Simplified: use dominant model from posterior
                        _bma_dom = max(_bma_weights_arr, key=_bma_weights_arr.get)
                        _bma_conc = float(_bma_weights_arr.get(_bma_dom, 0.0))
                        _signal_meta["bma_attribution"] = {
                            "dominant_model": str(_bma_dom),
                            "concentration": _bma_conc,
                        }
            except Exception:
                pass

        # Story 30.2: Gap-aware predict
        if GAP_AWARE_PREDICT_AVAILABLE and tuned_params:
            try:
                _tp_global = tuned_params.get("global", {})
                _gap_mu = _tp_global.get("mu", 0.0)
                _gap_P = _tp_global.get("P", 1e-4)
                _gap_phi = _tp_global.get("phi", 0.0)
                _gap_q = _tp_global.get("q", 1e-5)
                # Detect gap: check if last two trading dates are >1 calendar day apart
                _gap_days = 0
                if px is not None and len(px) >= 2:
                    _idx = px.index
                    if hasattr(_idx[-1], 'date'):
                        _delta = (_idx[-1] - _idx[-2]).days
                        _gap_days = max(0, _delta - 1)  # Weekends/holidays
                if _gap_days > 0:
                    _gap = gap_aware_predict(_gap_mu, _gap_P, _gap_phi, _gap_q, _gap_days)
                    _signal_meta["gap_aware"] = {
                        "gap_days": int(_gap_days),
                        "mu_predicted": float(_gap.mu_predicted) if hasattr(_gap, 'mu_predicted') else 0.0,
                        "P_predicted": float(_gap.P_predicted) if hasattr(_gap, 'P_predicted') else 0.0,
                    }
            except Exception:
                pass

        return {
            "status": "success",
            "asset": asset,
            "canon": canon,
            "title": title,
            "px": px,
            "feats": feats,
            "sigs": sigs,
            "thresholds": thresholds,
            "diagnostics": diagnostics,
            "last_close": last_close,
            "enrichment": enrichment,
            "data_quality": data_quality_meta,
            "signal_meta": _signal_meta,
        }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "asset": asset,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
