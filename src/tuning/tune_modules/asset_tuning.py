"""
Asset tuning: tune_asset_with_bma, _tune_worker.

Extracted from tune.py (Story 4.3). Contains the main per-asset BMA tuning
pipeline that orchestrates data loading, volatility fitting, regime detection,
and model averaging, plus the multiprocessing worker wrapper.
"""
import os
import datetime
import traceback
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor, as_completed

from ingestion.data_utils import fetch_px, _download_prices, get_default_asset_universe
from ingestion.adaptive_quality import adaptive_data_quality

from tuning.tune_modules.config import *  # noqa: F401,F403
from tuning.tune_modules.utilities import *  # noqa: F401,F403


__all__ = [
    "tune_asset_with_bma",
    "_tune_worker",
]


def tune_asset_with_bma(
    asset: str,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
    lambda_regime: float = 0.05,
    temporal_alpha: float = DEFAULT_TEMPORAL_ALPHA,
    previous_posteriors: Optional[Dict[int, Dict[str, float]]] = None,
    model_selection_method: str = DEFAULT_MODEL_SELECTION_METHOD,
    bic_weight: float = DEFAULT_BIC_WEIGHT,
) -> Optional[Dict]:
    """
    Tune asset parameters using full Bayesian Model Averaging.
    
    This is the upgraded entry point that implements:
    
        p(r_{t+H} | r) = Σ_m p(r_{t+H} | r, m, θ_{r,m}) · p(m | r)
    
    For EACH regime r:
    - Fits ALL candidate model classes independently
    - Computes model posteriors with temporal smoothing
    - Uses robust Hyvärinen score for model selection (optional)
    - Preserves full uncertainty across models
    
    NEVER selects a single best model — maintains full posterior.
    
    Args:
        asset: Asset symbol
        start_date: Start date for data
        end_date: End date (default: today)
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        lambda_regime: Hierarchical shrinkage strength
        temporal_alpha: Smoothing exponent for model posteriors
        previous_posteriors: Previous model posteriors per regime (for smoothing)
        model_selection_method: 'bic', 'hyvarinen', or 'combined' (default)
        bic_weight: Weight for BIC in combined method (0-1, default 0.5)
        
    Returns:
        Dictionary with structure:
        {
            "asset": str,
            "global": {
                "model_posterior": { m: p(m) },
                "models": { m: {...} }
            },
            "regime": {
                r: {
                    "model_posterior": { m: p(m|r) },
                    "models": { m: {...} },
                    "regime_meta": {...}
                }
            },
            "regime_counts": {...},
            "meta": {...},
            "timestamp": str
        }
    """
    # Minimum data thresholds
    MIN_DATA_FOR_REGIME = 100
    MIN_DATA_FOR_GLOBAL = 20
    
    # Reset filter cache for this asset to avoid cross-asset contamination
    # and collect fresh statistics for this tuning run
    if FILTER_CACHE_AVAILABLE:
        clear_filter_cache()
        reset_cache_stats()
    
    try:
        # Fetch price data (need OHLC for Garman-Klass volatility)
        # Use _download_prices ONCE to get OHLC data - extract Close from it
        # This avoids duplicate downloads (February 2026 optimization)
        df = _download_prices(asset, start_date, end_date)
        if df is None or df.empty:
            _log(f"     ⚠️  No price data for {asset}")
            return None
        
        # Adaptive data quality filter (February 2026)
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
        
        n_points = len(px) if px is not None else 0
        
        # For very small datasets, fall back directly to global-only tuning
        if n_points < MIN_DATA_FOR_GLOBAL:
            _log(f"     ⚠️  Insufficient data for {asset} ({n_points} points) - need at least {MIN_DATA_FOR_GLOBAL}")
            return None
        
        # For small-to-medium datasets (20-100 points), skip regime tuning but do global
        if n_points < MIN_DATA_FOR_REGIME:
            _log(f"     ⚠️  Insufficient data for {asset} ({n_points} points) for regime tuning")
            _log(f"     ↩️  Falling back to global-only model tuning...")
            
            # Do global tuning only
            global_result = tune_asset_q(
                asset=asset,
                start_date=start_date,
                end_date=end_date,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda
            )
            
            if global_result is None:
                _log(f"     ⚠️  Global tuning also failed for {asset}")
                return None
            
            # Return result with explicit markers that regime tuning was skipped
            # Note: tune_asset_q now returns {"has_bma": True, "global": {...}}
            # We need to extract the inner global data
            global_data = global_result.get('global', global_result)
            return {
                "asset": asset,
                "has_bma": True,  # CRITICAL: signals.py checks this flag to accept the cache
                "global": global_data,
                "regime": None,  # Explicitly None - no regime params available
                "use_regime_tuning": False,
                "regime_fallback": True,
                "regime_fallback_reason": f"insufficient_data_for_regime_bma",
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

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

        # Story 7.3: Inflate vol on overnight gap days BEFORE model fitting.
        # Gaps cause the Kalman filter to over-react (treats gap as drift change).
        # Inflating obs variance on gap days makes the filter more honest.
        if VOL_FUSION_AVAILABLE and open_ is not None and len(open_) >= min_len:
            try:
                _gap_close = close[:min_len] if close is not None else None
                _gap_open = open_[:min_len]
                if _gap_close is not None:
                    _gap_result = detect_overnight_gap(_gap_open, _gap_close, vol=vol)
                    if _gap_result.n_gaps > 0:
                        _gap_var = _gap_result.gap_magnitude ** 2 / 4.0
                        vol = np.where(_gap_result.is_gap, np.sqrt(vol ** 2 + _gap_var), vol)
                        _log(f"     Gap vol inflate: {_gap_result.n_gaps} gap days ({100*_gap_result.gap_fraction:.1f}%)")
            except Exception:
                pass

        # Remove NaN/Inf and stale-price observations (zero-return days)
        # Also filter Volume=0 phantom quotes (February 2026)
        _STALE_RETURN_THRESHOLD = 1e-10
        valid_mask = (np.isfinite(returns) & np.isfinite(vol) & (vol > 0)
                      & (np.abs(returns) > _STALE_RETURN_THRESHOLD))
        # Add Volume >= 100 filter if Volume data available (February 2026)
        # Volume < 100 indicates phantom OTC quotes without genuine price discovery
        # Skip for FX pairs (=X) and indices (^) — Yahoo reports Volume=0
        _MIN_GENUINE_VOLUME = 100
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

        # After cleaning, check if we still have enough data for regime tuning
        if len(returns) < MIN_DATA_FOR_REGIME:
            if len(returns) < MIN_DATA_FOR_GLOBAL:
                _log(f"     ⚠️  Insufficient valid data for {asset} after cleaning ({len(returns)} returns)")
                return None
            
            _log(f"     ⚠️  Insufficient data for {asset} after cleaning ({len(returns)} returns) for regime tuning")
            _log(f"     ↩️  Falling back to global-only model tuning...")
            
            # Do global tuning only
            global_result = tune_asset_q(
                asset=asset,
                start_date=start_date,
                end_date=end_date,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda
            )
            
            if global_result is None:
                _log(f"     ⚠️  Global tuning also failed for {asset}")
                return None
            
            # Note: tune_asset_q now returns {"has_bma": True, "global": {...}}
            # We need to extract the inner global data
            global_data = global_result.get('global', global_result)
            return {
                "asset": asset,
                "has_bma": True,  # CRITICAL: signals.py checks this flag to accept the cache
                "global": global_data,
                "regime": None,
                "use_regime_tuning": False,
                "regime_fallback": True,
                "regime_fallback_reason": f"insufficient_data_after_cleaning",
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

        # Assign regime labels (with optional computation cache)
        _log(f"     Assigning regime labels for {len(returns)} observations...")
        if COMPUTATION_CACHE_AVAILABLE and _computation_cache is not None:
            regime_labels = _computation_cache.get_regime(
                asset, returns, lambda r: assign_regime_labels(r, vol)
            )
        else:
            regime_labels = assign_regime_labels(returns, vol)

        # Count samples per regime
        regime_counts = {r: int(np.sum(regime_labels == r)) for r in range(5)}
        _log(f"     Regime distribution: " + ", ".join([f"{REGIME_LABELS[r]}={c}" for r, c in sorted(regime_counts.items()) if c > 0]))

        # First get global params (for backward compatibility)
        _log(f"     🔧 Estimating global parameters...")
        global_result = tune_asset_q(
            asset=asset,
            start_date=start_date,
            end_date=end_date,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )

        if global_result is None:
            return None

        # =================================================================
        # BAYESIAN MODEL AVERAGING: Fit ALL models for each regime
        # =================================================================
        # This implements the governing law:
        #     p(r_{t+H} | r) = Σ_m p(r_{t+H} | r, m, θ_{r,m}) · p(m | r)
        # =================================================================
        _log(f"     🔄 Bayesian Model Averaging (λ_regime={lambda_regime})...")
        if previous_posteriors is not None:
            _log(f"        ↪ Using previous posteriors for temporal smoothing (α={DEFAULT_TEMPORAL_ALPHA})")
        
        # Extract prices array for MR integration (February 2026)
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
        
        # Compute GK c prior from OHLC data (Story 2.2)
        _gk_c_prior_bma = None
        if GK_C_PRIOR_AVAILABLE:
            try:
                _gk_c_prior_bma = gk_c_prior(open_, high, low, close)
            except Exception:
                _gk_c_prior_bma = None

        bma_result = tune_regime_model_averaging(
            returns=returns,
            vol=vol,
            regime_labels=regime_labels,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda,
            min_samples=MIN_REGIME_SAMPLES,
            temporal_alpha=DEFAULT_TEMPORAL_ALPHA,
            previous_posteriors=previous_posteriors,  # Use provided previous posteriors
            lambda_regime=lambda_regime,
            prices=prices_array,  # MR integration (February 2026)
            asset=asset,  # FIX #4: Asset-class adaptive c bounds
            gk_c_prior_value=_gk_c_prior_bma,  # Story 2.2
        )

        # Collect diagnostics summary
        regime_results = bma_result.get("regime", {})
        n_active = sum(1 for r, p in regime_results.items() 
                       if not p.get("regime_meta", {}).get("fallback", False))
        n_shrunk = sum(1 for r, p in regime_results.items() 
                       if p.get("regime_meta", {}).get("shrinkage_applied", False))
        collapse_warnings = 0
        for r_data in regime_results.values():
            if r_data.get("regime_meta", {}).get("collapse_warning", False):
                collapse_warnings += 1

        # Build combined result with BMA structure
        # Note: tune_asset_q now returns {"has_bma": True, "global": {...}}
        # We need to extract the inner global data
        global_data = global_result.get('global', global_result)  # Backward compatible
        
        result = {
            "asset": asset,
            "has_bma": True,  # CRITICAL: signals.py checks this flag to accept the cache
            "global": {
                # Keep backward-compatible global result
                **global_data,
                # Add BMA global model posterior
                "model_posterior": bma_result.get("global", {}).get("model_posterior", {}),
                "models": bma_result.get("global", {}).get("models", {}),
                # Volatility estimator used (February 2026)
                "volatility_estimator": vol_estimator_used,
                # Market conditioning flags (February 2026)
                "market_conditioning_enabled": MARKET_CONDITIONING_ENABLED and MARKET_CONDITIONING_AVAILABLE,
                "vix_nu_adjustment_enabled": MARKET_CONDITIONING_ENABLED and MARKET_CONDITIONING_AVAILABLE and global_data.get("nu") is not None,
            },
            "regime": regime_results,  # Now contains model_posterior and models per regime
            "use_regime_tuning": True,
            "regime_counts": regime_counts,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hierarchical_tuning": {
                "lambda_regime": lambda_regime,
                "temporal_alpha": DEFAULT_TEMPORAL_ALPHA,
                "n_active_regimes": n_active,
                "n_shrunk_regimes": n_shrunk,
                "collapse_warning": collapse_warnings,
            },
            "meta": bma_result.get("meta", {}),
        }

        # ─────────────────────────────────────────────────────────────
        # CALIBRATION PARAMS PROMOTION (March 2026)
        # Extract BMA-weighted calibration params from per-model results
        # and promote to global level for signals.py consumption.
        #
        # Uses BMA-weighted aggregation across all models that produced
        # calibration params. This ensures the correction reflects the
        # model ensemble, not just a single model.
        # ─────────────────────────────────────────────────────────────
        try:
            _bma_models = bma_result.get("global", {}).get("models", {})
            _bma_posterior = bma_result.get("global", {}).get("model_posterior", {})
            _agg_cal = {}
            _total_w = 0.0
            for _m_name, _m_data in _bma_models.items():
                _m_cal = _m_data.get("calibration_params", {})
                _m_w = _bma_posterior.get(_m_name, 0.0)
                if _m_cal and _m_w > 0:
                    _total_w += _m_w
                    for _k, _v in _m_cal.items():
                        if isinstance(_v, (int, float)):
                            _agg_cal[_k] = _agg_cal.get(_k, 0.0) + _m_w * _v
                        elif isinstance(_v, list) and _k not in _agg_cal:
                            # For isotonic knots, take from highest-weight model
                            _agg_cal[_k] = _v
            # Normalize weighted averages
            if _total_w > 0:
                for _k in _agg_cal:
                    if isinstance(_agg_cal[_k], (int, float)):
                        _agg_cal[_k] = _agg_cal[_k] / _total_w
            if _agg_cal:
                result["global"]["calibration_params"] = _agg_cal
        except Exception:
            pass  # Calibration param promotion is best-effort

        # =================================================================
        # POST-FIT DIAGNOSTICS (Tune.md Story Integration)
        # =================================================================
        if "diagnostics" not in result:
            result["diagnostics"] = {}

        _gd = result.get("global", {})
        _diag_q = _gd.get("q", 1e-5)
        _diag_c = _gd.get("c", 1.0)
        _diag_phi = _gd.get("phi", 0.0)
        _diag_nu = _gd.get("nu")

        # Story 2.1: Regime-conditional observation noise
        if REGIME_C_AVAILABLE:
            try:
                _rc = fit_regime_c(
                    returns, vol, regime_labels,
                    q=_diag_q, phi=_diag_phi, c_scalar=_diag_c,
                    nu=_diag_nu,
                )
                _c_per_regime_dict = {}
                if hasattr(_rc, 'c_per_regime') and _rc.c_per_regime is not None:
                    if isinstance(_rc.c_per_regime, np.ndarray):
                        _c_per_regime_dict = {str(i): float(v) for i, v in enumerate(_rc.c_per_regime)}
                    elif isinstance(_rc.c_per_regime, dict):
                        _c_per_regime_dict = {str(k): float(v) for k, v in _rc.c_per_regime.items()}
                result["diagnostics"]["regime_c"] = {
                    "c_per_regime": _c_per_regime_dict,
                    "delta_bic": float(_rc.delta_bic) if hasattr(_rc, 'delta_bic') else 0.0,
                    "fit_success": bool(_rc.fit_success) if hasattr(_rc, 'fit_success') else False,
                }
            except Exception:
                pass

        # Story 3.2: Rolling phi estimation
        if ROLLING_PHI_AVAILABLE:
            try:
                _rp = rolling_phi_estimate(returns, vol, asset_symbol=asset)
                result["diagnostics"]["rolling_phi"] = {
                    "phi_mean": float(_rp.phi_mean),
                    "phi_std": float(_rp.phi_std),
                    "n_breaks": int(_rp.n_breaks),
                }
            except Exception:
                pass

        # Story 3.3: Phi-nu identifiability check
        if PHI_NU_IDENTIFIABILITY_AVAILABLE and _diag_nu is not None:
            try:
                _ident = check_phi_nu_identifiability(
                    returns, vol, _diag_q, _diag_c, _diag_phi, _diag_nu,
                    asset_symbol=asset,
                )
                result["diagnostics"]["phi_nu_identifiability"] = {
                    "condition_number": float(_ident.condition_number),
                    "is_critical": bool(_ident.is_critical),
                    "regularization_applied": bool(_ident.regularization_applied),
                }
                # When condition number is critical, write regularized values back
                # to the model. This prevents unreliable phi/nu from reaching inference.
                if _ident.regularization_applied and _ident.is_critical:
                    _gd["phi"] = _ident.phi_regularized
                    _gd["nu"] = _ident.nu_regularized
                    result["diagnostics"]["phi_nu_identifiability"]["phi_original"] = float(_diag_phi)
                    result["diagnostics"]["phi_nu_identifiability"]["nu_original"] = float(_diag_nu)
                    result["diagnostics"]["phi_nu_identifiability"]["phi_regularized"] = float(_ident.phi_regularized)
                    result["diagnostics"]["phi_nu_identifiability"]["nu_regularized"] = float(_ident.nu_regularized)
            except Exception:
                pass

        # Story 7.1-7.3: Vol fusion, HAR-GK hybrid, overnight gap detection
        # These require OHLC data (open_, high, low, close) which are available
        # when GK_VOLATILITY_AVAILABLE is True (the normal path).
        _has_ohlc = 'open_' in dir() and open_ is not None
        if VOL_FUSION_AVAILABLE and _has_ohlc:
            try:
                _vf = vol_fusion_kernel(
                    open_=open_, high=high, low=low, close=close,
                    returns=returns, regime=regime_labels,
                )
                result["diagnostics"]["vol_fusion"] = {
                    "volatility_last": float(_vf.volatility[-1]) if len(_vf.volatility) > 0 else None,
                    "method": _vf.method,
                }
            except Exception:
                pass
            try:
                _hgk = har_gk_hybrid(
                    open_=open_, high=high, low=low, close=close,
                )
                result["diagnostics"]["har_gk_hybrid"] = {
                    "hybrid_vol_last": float(_hgk.volatility[-1]) if len(_hgk.volatility) > 0 else None,
                    "weights": _hgk.weights.tolist() if _hgk.weights is not None else None,
                    "weights_method": _hgk.weights_method,
                }
            except Exception:
                pass
            try:
                _og = detect_overnight_gap(
                    open_=open_, close=close, vol=vol,
                )
                result["diagnostics"]["overnight_gaps"] = {
                    "n_gaps": int(_og.n_gaps),
                    "gap_fraction": float(_og.gap_fraction),
                }
            except Exception:
                pass

        # Story 8.1: Continuous nu refinement
        if CONTINUOUS_NU_AVAILABLE and _diag_nu is not None:
            try:
                _cnr = refine_nu_continuous(returns, vol, _diag_q, _diag_c, _diag_phi, _diag_nu)
                result["diagnostics"]["continuous_nu"] = {
                    "nu_refined": float(_cnr.nu_refined),
                    "bic_improvement": float(_cnr.bic_improvement),
                }
                # Apply refined nu to global if improvement is meaningful
                if _cnr.bic_improvement > 2.0:
                    result["global"]["nu_refined"] = float(_cnr.nu_refined)
                    result["global"]["nu_refinement_bic_delta"] = float(_cnr.bic_improvement)
            except Exception:
                pass

        # Story 8.3: VIX-conditional nu
        # vix_conditional_nu(nu_base, vix_current) adjusts nu based on VIX level.
        # Fetch current VIX from market_conditioning module.
        if VIX_CONDITIONAL_NU_AVAILABLE and _diag_nu is not None:
            try:
                _vix_current = None
                try:
                    from calibration.market_conditioning import get_current_vix
                    _vix_current = get_current_vix()
                except Exception:
                    pass
                if _vix_current is not None:
                    _vcn = vix_conditional_nu(_diag_nu, _vix_current)
                    result["diagnostics"]["vix_conditional_nu"] = {
                        "nu_base": float(_diag_nu),
                        "nu_adjusted": float(_vcn),
                        "vix_current": float(_vix_current),
                    }
            except Exception:
                pass

        # Story 9.2-9.3: Innovation diagnostics
        if INNOVATION_DIAGNOSTICS_AVAILABLE:
            try:
                from models.gaussian import kalman_filter_drift_phi
                _mu_f, _P_f, _ = kalman_filter_drift_phi(
                    returns, vol, _diag_q, _diag_c, _diag_phi,
                )
                _innovations = returns - _mu_f
                _R_diag = _diag_c * vol ** 2
                _vr = innovation_variance_ratio(_innovations, _R_diag)
                _cs = innovation_cusum(_innovations, _R_diag)
                result["diagnostics"]["innovation_variance_ratio"] = float(_vr.current_vr)
                result["diagnostics"]["innovation_cusum_max"] = float(_cs.max_cusum)
                result["diagnostics"]["innovation_cusum_alert"] = _cs.alert
                # Story 9.2: Apply c correction when VR indicates miscalibration.
                # c_correction is sqrt-dampened: c_new = c_old * VR^0.5
                if _vr.needs_correction:
                    _c_corrected = _diag_c * _vr.c_correction
                    _gd["c"] = float(_c_corrected)
                    result["diagnostics"]["innovation_c_corrected"] = True
                    result["diagnostics"]["innovation_c_original"] = float(_diag_c)
                    result["diagnostics"]["innovation_c_new"] = float(_c_corrected)
            except Exception:
                pass

        # Story 19.1-19.3: Regime classification diagnostics
        if REGIME_CLASSIFICATION_AVAILABLE:
            try:
                _last_vol = float(vol[-1]) if len(vol) > 0 else 0.01
                _last_drift = float(returns[-20:].mean()) if len(returns) >= 20 else 0.0
                _med_vol = float(np.median(vol)) if len(vol) > 0 else 0.01
                _srm = soft_regime_membership(_last_vol, _last_drift, _med_vol)
                result["diagnostics"]["soft_regime_membership"] = {
                    "p_high_vol": float(_srm.p_high_vol) if hasattr(_srm, 'p_high_vol') else 0.0,
                    "p_trending": float(_srm.p_trending) if hasattr(_srm, 'p_trending') else 0.0,
                }
            except Exception:
                pass
            try:
                _hmm = hmm_regime_fit(vol, returns)
                result["diagnostics"]["hmm_regime"] = {
                    "n_regimes": int(_hmm.n_regimes) if hasattr(_hmm, 'n_regimes') else 0,
                    "converged": bool(_hmm.converged) if hasattr(_hmm, 'converged') else False,
                }
            except Exception:
                pass

        # Story 20.1-20.2: Multi-scale kappa & equilibrium shift
        if OU_MEAN_REVERSION_AVAILABLE:
            try:
                _prices_arr = prices if prices is not None else returns.cumsum()
                _msk = multi_scale_kappa(np.asarray(_prices_arr))
                result["diagnostics"]["multi_scale_kappa"] = {
                    "pooled_kappa": float(_msk.pooled_kappa) if hasattr(_msk, 'pooled_kappa') else 0.0,
                    "half_life": float(np.log(2) / max(_msk.pooled_kappa, 1e-10)) if hasattr(_msk, 'pooled_kappa') else None,
                }
            except Exception:
                pass
            try:
                from models.gaussian import kalman_filter_drift_phi
                _mu_smooth, _, _ = kalman_filter_drift_phi(
                    returns, vol, _diag_q, _diag_c, _diag_phi,
                )
                _eqs = detect_equilibrium_shift(_mu_smooth)
                result["diagnostics"]["equilibrium_shifts"] = {
                    "n_changepoints": int(_eqs.n_changepoints) if hasattr(_eqs, 'n_changepoints') else 0,
                }
            except Exception:
                pass

        # Story 21.1-21.3: RTS smoother & EM parameter update
        if RTS_SMOOTHER_AVAILABLE:
            try:
                from models.gaussian import kalman_filter_drift_phi
                _mu_f, _P_f, _ = kalman_filter_drift_phi(
                    returns, vol, _diag_q, _diag_c, _diag_phi,
                )
                # Compute predicted states for RTS smoother
                _T = len(_mu_f)
                _mu_pred = np.empty(_T)
                _P_pred = np.empty(_T)
                _mu_pred[0] = _mu_f[0]
                _P_pred[0] = _P_f[0] + _diag_q
                for _t in range(1, _T):
                    _mu_pred[_t] = _diag_phi * _mu_f[_t - 1]
                    _P_pred[_t] = _diag_phi ** 2 * _P_f[_t - 1] + _diag_q
                _rts = rts_smoother_backward(_mu_f, _P_f, _mu_pred, _P_pred, _diag_phi, _diag_q)
                _mu_s = _rts.mu_smooth if hasattr(_rts, 'mu_smooth') else _rts[0]
                _P_s = _rts.P_smooth if hasattr(_rts, 'P_smooth') else _rts[1]
                _em_q, _em_c, _em_phi = em_parameter_update(
                    returns, vol, _mu_s, _P_s, _diag_phi,
                )
                result["diagnostics"]["rts_smoother"] = {
                    "em_q": float(_em_q),
                    "em_c": float(_em_c),
                    "em_phi": float(_em_phi),
                }
                _sm_innov = smoothed_innovations(returns, _mu_s, vol)
                result["diagnostics"]["smoothed_ljung_box_pvalue"] = float(_sm_innov.ljung_box_pvalue) if hasattr(_sm_innov, 'ljung_box_pvalue') else float(_sm_innov.pvalue) if hasattr(_sm_innov, 'pvalue') else None
            except Exception:
                pass

        # Print summary
        global_posterior = result["global"].get("model_posterior", {})
        if global_posterior:
            posterior_str = ", ".join([f"{m}={p:.3f}" for m, p in global_posterior.items()])
            _log(f"     ✓ Global model posterior: {posterior_str}")
        else:
            _log(f"     ✓ Global: q={global_result['q']:.2e}, φ={global_result.get('phi', 'N/A')}")
        
        for r, r_data in regime_results.items():
            regime_meta = r_data.get("regime_meta", {})
            if not regime_meta.get("fallback", False):
                model_posterior = r_data.get("model_posterior", {})
                posterior_str = ", ".join([f"{m}={p:.3f}" for m, p in model_posterior.items()])
                shrunk_marker = " [shrunk]" if regime_meta.get("shrinkage_applied", False) else ""
                _log(f"     ✓ {REGIME_LABELS[int(r)]}: {posterior_str}{shrunk_marker}")

        if collapse_warnings > 0:
            _log(f"     ⚠️  Collapse warnings: regime parameters too close to global")

        # Report filter cache statistics
        if FILTER_CACHE_AVAILABLE:
            cache_stats = get_cache_stats()
            if cache_stats.hits > 0 or cache_stats.misses > 0:
                _log(f"     📊 {cache_stats.summary()}")
            # Add cache stats to result for analysis
            result["filter_cache_stats"] = {
                "hits": cache_stats.hits,
                "misses": cache_stats.misses,
                "hit_rate": cache_stats.hit_rate,
                "fold_slice_reuses": cache_stats.fold_slice_reuses,
                "warm_starts": cache_stats.warm_starts,
            }

        return result

    except Exception as e:
        import traceback
        _log(f"     ❌ {asset}: Failed - {e}")
        traceback.print_exc()
        raise  # Re-raise so caller can handle it


def _tune_worker(args_tuple: Tuple[str, str, Optional[str], float, float, float, Optional[Dict]]) -> Tuple[str, Optional[Dict], Optional[str], Optional[str]]:
    """
    Worker function for parallel asset tuning.
    Must be defined at module level for ProcessPoolExecutor pickling.
    
    Args:
        args_tuple: (asset, start_date, end_date, prior_log_q_mean, prior_lambda, lambda_regime, previous_posteriors)
        
    Returns:
        Tuple of (asset, result_dict, error_message, traceback_str)
        - If success: (asset, result, None, None)
        - If failure: (asset, None, error_string, traceback_string)
    """
    asset, start_date, end_date, prior_log_q_mean, prior_lambda, lambda_regime, previous_posteriors = args_tuple
    
    # Track failure reasons for better error reporting
    failure_reasons = []
    
    try:
        result = tune_asset_with_bma(
            asset=asset,
            start_date=start_date,
            end_date=end_date,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda,
            lambda_regime=lambda_regime,
            previous_posteriors=previous_posteriors,
        )
        
        if result:
            return (asset, result, None, None)
        
        failure_reasons.append("tune_asset_with_bma returned None (likely insufficient data or data fetch error)")

        # Fallback to standard tuning when regime tuning fails (insufficient data for regime estimation)
        _log(f"  ↩️  {asset}: Falling back to standard model tuning...")
        fallback_result = tune_asset_q(
            asset=asset,
            start_date=start_date,
            end_date=end_date,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )

        if fallback_result:
            # tune_asset_q now returns BMA-compatible structure with global wrapper
            # Just mark as fallback for diagnostic purposes
            fallback_result['regime_fallback'] = True
            fallback_result['regime_fallback_reason'] = 'tune_asset_with_bma_returned_none'
            return (asset, fallback_result, None, None)
        else:
            failure_reasons.append("tune_asset_q also returned None")
            
            # Try to get more info about why data fetch might have failed
            try:
                df = _download_prices(asset, start_date, end_date)
                if df is None:
                    failure_reasons.append(f"_download_prices returned None for {asset}")
                elif df.empty:
                    failure_reasons.append(f"_download_prices returned empty DataFrame for {asset}")
                else:
                    n_rows = len(df)
                    failure_reasons.append(f"Data was fetched ({n_rows} rows) but processing failed")
                    # Check for NaN/Inf issues
                    if 'Close' in df.columns:
                        close = df['Close']
                        n_valid = close.notna().sum()
                        n_inf = np.isinf(close.replace([np.inf, -np.inf], np.nan).dropna()).sum() if n_valid > 0 else 0
                        failure_reasons.append(f"Close prices: {n_valid} valid, {n_rows - n_valid} NaN, {n_inf} Inf")
            except Exception as data_check_err:
                failure_reasons.append(f"Data check error: {data_check_err}")
            
            detailed_error = " | ".join(failure_reasons)
            return (asset, None, detailed_error, None)

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        return (asset, None, str(e), tb_str)


