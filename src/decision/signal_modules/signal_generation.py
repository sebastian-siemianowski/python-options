"""Signal generation engine: latest_signals().

Extracted from signals.py - Story 8.4.
Contains the main latest_signals() function (~1,434 lines) that computes
signal objects from features, MC simulations, and calibration.
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
from decision.signal_modules.regime_classification import (  # Private names
    _SIG_H_ANNUAL_CAP, _CI_LOG_FLOOR, _CI_LOG_CAP,
    _DISPLAY_PRICE_CACHE, _compute_sig_h_cap, _smooth_display_price,
    _logistic, _CUSUM_STATE, _get_cusum_state,
)
from decision.signal_modules.data_fetching import *  # noqa: F403
from decision.signal_modules.momentum_features import *  # noqa: F403
from decision.signal_modules.kalman_diagnostics import *  # noqa: F403
from decision.signal_modules.kalman_diagnostics import (  # Private names
    _test_innovation_whiteness, _compute_kalman_log_likelihood,
    _compute_kalman_log_likelihood_heteroskedastic, _estimate_regime_drift_priors,
)
from decision.signal_modules.parameter_loading import *  # noqa: F403
from decision.signal_modules.parameter_loading import (  # Private names
    _safe_get_nested, _load_tuned_kalman_params, _select_regime_params,
)
from decision.signal_modules.kalman_filtering import *  # noqa: F403
from decision.signal_modules.kalman_filtering import (  # Private names
    _compute_kalman_gain_from_filtered, _apply_gain_monitoring_reset,
    _kalman_filter_drift,
)
from decision.signal_modules.feature_pipeline import *  # noqa: F403
from decision.signal_modules.hmm_regimes import *  # noqa: F403
from decision.signal_modules.monte_carlo import *  # noqa: F403
from decision.signal_modules.monte_carlo import _simulate_forward_paths  # noqa: F401
from decision.signal_modules.bma_engine import *  # noqa: F403
from decision.signal_modules.threshold_calibration import *  # noqa: F403
from decision.signal_modules.probability_mapping import *  # noqa: F403
from decision.signal_modules.probability_mapping import (  # Private names
    _load_signals_calibration, _apply_single_p_map, _apply_p_up_calibration,
    _apply_emos_correction, _apply_magnitude_bias_correction,
    _get_calibrated_label_thresholds,
)
from decision.signal_modules.signal_dataclass import *  # noqa: F403
from ingestion.data_utils import _ensure_float_series  # noqa: F401

# ---------------------------------------------------------------------------
from decision.signal_modules.signal_state import (  # noqa: E402
    load_signal_state,
    save_signal_state,
    SIGNAL_STATE_DIR,
    SIGNAL_STATE_DEFAULT_P,
)

# ---------------------------------------------------------------------------
# Globals needed by latest_signals (originally in signals.py module scope)
# ---------------------------------------------------------------------------
NOTIONAL_PLN = 1_000_000

MOM_DRIFT_SCALE = 0.10
MOM_CROSSOVER_HORIZON = 63
MOM_MIN_OBSERVATIONS = 126
MOMENTUM_HALF_LIFE_DAYS = 42
MOM_SLOW_FRAC = 0.30

QUANTILE_CI_MIN_SAMPLES = 100

RETURN_CAP_BY_CLASS: Dict[str, float] = {
    "equity": 0.30,
    "currency": 0.15,
    "metal": 0.20,
    "crypto": 1.00,
    "etf": 0.25,
}
RETURN_CAP_DEFAULT = 0.30

def latest_signals(feats: Dict[str, pd.Series], horizons: List[int], last_close: float, t_map: bool = True, ci: float = 0.68, tuned_params: Optional[Dict] = None, asset_key: Optional[str] = None, _calibration_fast_mode: bool = True, n_mc_paths: int = 10000, asset_type: str = "equity") -> Tuple[List[Signal], Dict[int, Dict[str, float]]]:
    """Compute signals using regime-aware priors, tail-aware probability mapping, and
    anti-snap logic (two-day confirmation + hysteresis + smoothing) without extra flags.

    Uses Bayesian Model Averaging when tuned_params with BMA structure is provided.

    Story 3.1: Coherent multi-horizon MC is now the default (_calibration_fast_mode=True).
    A single MC call produces all horizons from the same paths, ensuring monotonic
    variance and coherent cross-horizon signals.

    CONTRACT WITH tune.py:
    - tuned_params contains model_posterior and models for each regime
    - Tune guarantees non-empty outputs via hierarchical borrowing
    - This function does NOT implement tuning or fallback logic

    Args:
        feats: Feature dictionary from compute_features()
        horizons: List of forecast horizons in days
        last_close: Last close price
        t_map: Whether to use Student-t probability mapping
        ci: Confidence interval for bands
        tuned_params: Full tuned params from _load_tuned_kalman_params() with BMA structure
        asset_key: Optional asset identifier for display price inertia (Upgrade #3)

    We build last‑two‑days estimates to avoid look‑ahead while adding stability.
    """
    idx = feats.get("px", pd.Series(dtype=float)).index
    if idx is None or len(idx) < 2:
        # Fallback to simple single‑day path
        idx = pd.DatetimeIndex(idx)
    last2 = idx[-2:] if len(idx) >= 2 else idx

    # Helper to safely fetch last/prev values from a Series
    def _tail2(series_key: str, default_val: float = np.nan) -> Tuple[float, float]:
        s = feats.get(series_key, None)
        if s is None or not isinstance(s, pd.Series) or s.empty:
            return (default_val, default_val)
        s2 = s.reindex(last2)
        vals = s2.to_numpy(dtype=float)
        if vals.size == 1:
            return (float(vals[-1]), float(vals[-1]))
        return (float(vals[-1]), float(vals[-2]))

    # Prefer posterior drift if available
    mu_now, mu_prev = _tail2("mu_post", 0.0)
    if not np.isfinite(mu_now):
        mu_now = 0.0
    if not np.isfinite(mu_prev):
        mu_prev = mu_now
    vol_now, vol_prev = _tail2("vol", np.nan)
    vol_reg_now, vol_reg_prev = _tail2("vol_regime", 1.0)
    trend_now, trend_prev = _tail2("trend_z", 0.0)
    z5_now, z5_prev = _tail2("z5", 0.0)
    # Use globally-fitted nu_hat (Level-7); fall back to rolling nu if missing/invalid
    nu_hat_series = feats.get("nu_hat")
    if isinstance(nu_hat_series, pd.Series) and not nu_hat_series.empty:
        nu_glob = float(nu_hat_series.iloc[-1])
    else:
        nu_glob, _ = _tail2("nu", 50.0)
        if not np.isfinite(nu_glob):
            nu_glob = 50.0
    nu_glob = float(np.clip(nu_glob, 4.5, 500.0))
    # Prefer smoothed skew if available; fallback to raw skew; default neutral 0.0
    skew_now, skew_prev = _tail2("skew_s", np.nan)
    if not np.isfinite(skew_now) or not np.isfinite(skew_prev):
        skew_now_fallback, skew_prev_fallback = _tail2("skew", 0.0)
        if not np.isfinite(skew_now):
            skew_now = skew_now_fallback
        if not np.isfinite(skew_prev):
            skew_prev = skew_prev_fallback

    moms_now = [
        _tail2("mom21", 0.0)[0],
        _tail2("mom63", 0.0)[0],
        _tail2("mom126", 0.0)[0],
        _tail2("mom252", 0.0)[0],
    ]
    moms_prev = [
        _tail2("mom21", 0.0)[1],
        _tail2("mom63", 0.0)[1],
        _tail2("mom126", 0.0)[1],
        _tail2("mom252", 0.0)[1],
    ]

    # Mapping function that accepts per‑day skew/nu
    km_prob = (feats.get("kalman_metadata") or {})
    noise_model = km_prob.get("kalman_noise_model")
    tuned_nu_meta = km_prob.get("kalman_nu")
    is_student_world = noise_model and noise_model.startswith('phi_student_t_nu_')
    if is_student_world and (tuned_nu_meta is None or not np.isfinite(tuned_nu_meta)):
        raise ValueError("Student-t model selected but ν missing from tuning cache")
    nu_prob = float(tuned_nu_meta) if is_student_world else nu_glob

    # Check for GH (Generalized Hyperbolic) model - captures skewness
    is_gh_world = noise_model == 'generalized_hyperbolic'
    gh_params = km_prob.get("gh_model", {}).get("parameters", {}) if is_gh_world else {}
    gh_lambda = gh_params.get("lambda", -0.5)
    gh_alpha = gh_params.get("alpha", 1.0)
    gh_beta = gh_params.get("beta", 0.0)
    gh_delta = gh_params.get("delta", 1.0)

    # Import GH CDF function if GH model is used
    gh_cdf_func = None
    if is_gh_world:
        try:
            from calibration.gh_distribution import gh_cdf
            gh_cdf_func = gh_cdf
        except ImportError:
            # Fallback to Student-t if GH not available
            is_gh_world = False
            is_student_world = True
            nu_prob = 8.0  # Default to moderate tails

    # Mapping function that accepts per‑day skew/nu
    def map_prob(edge: float, nu_val: float, skew_val: float) -> float:
        if not np.isfinite(edge):
            return 0.5
        z = float(edge)
        nu_eff = nu_prob if is_student_world else nu_val

        # GH model: use fitted GH CDF for probability mapping
        if is_gh_world and gh_cdf_func is not None:
            try:
                # GH CDF returns P(Z <= z), we need this for probability computation
                base_p = float(gh_cdf_func(np.array([z]), gh_lambda, gh_alpha, gh_beta, gh_delta)[0])
                if not np.isfinite(base_p):
                    # Fallback to Student-t if GH fails
                    base_p = float(student_t.cdf(z, df=8.0))
            except Exception:
                base_p = float(student_t.cdf(z, df=8.0))
        # Base symmetric mapping dictated solely by model identity
        elif is_student_world:
            base_p = float(student_t.cdf(z, df=nu_eff))
        else:
            base_p = float(norm.cdf(z))
        if not np.isfinite(base_p):
            return 0.5
        # Edgeworth asymmetry using realized skew and (optional) kurt proxy from nu
        g1 = float(np.clip(skew_val if np.isfinite(skew_val) else 0.0, -1.5, 1.5))
        if np.isfinite(nu_val) and nu_val > 4.5 and nu_val < 1e9:
            g2 = 6.0 / (float(nu_val) - 4.0)
        else:
            g2 = 0.0
        if g1 == 0.0 and g2 == 0.0:
            return float(np.clip(base_p, 0.001, 0.999))
        try:
            phi = float(norm.pdf(z))
            corr = (g1 / 6.0) * (1.0 - z * z) + (g2 / 24.0) * (z ** 3 - 3.0 * z) - (g1 ** 2 / 36.0) * (2.0 * z ** 3 - 5.0 * z)
            # Damp skew influence in extreme tails to stabilize mapping for |z|>~3
            try:
                damp = math.exp(-0.5 * z * z)
            except Exception:
                damp = 0.0
            p = base_p + phi * (corr * damp)
            return float(np.clip(p, 0.001, 0.999))
        except Exception:
            return float(np.clip(base_p, 0.001, 0.999))

    # Regime detection via HMM posterior inference (replaces threshold-based heuristics)
    hmm_result = feats.get("hmm_result")
    reg, regime_meta = infer_current_regime(feats, hmm_result)

    # CI quantile based on 'now'
    alpha = np.clip(ci, 1e-6, 0.999999)
    tail = 0.5 * (1 + alpha)
    if is_gh_world:
        # For GH, use Student-t approximation for quantile (GH quantile is expensive)
        # The GH model captures skewness in CDF, but for CI we use symmetric approximation
        z_star = float(student_t.ppf(tail, df=8.0))
    elif is_student_world:
        z_star = float(student_t.ppf(tail, df=float(nu_prob)))
    else:
        z_star = float(norm.ppf(tail))

    # Median vol for uncertainty component (use rolling median series if available)
    vol_series = feats.get("vol", pd.Series(dtype=float))
    try:
        med_vol_series = vol_series.rolling(252, min_periods=63).median()
        med_vol_last = safe_last(med_vol_series) if med_vol_series is not None else float('nan')
    except Exception:
        med_vol_last = float('nan')
    # Explicit, readable fallback to long-run belief anchor (global median over entire history)
    if not np.isfinite(med_vol_last) or med_vol_last <= 0:
        try:
            med_vol_last = float(np.nanmedian(np.asarray(vol_series.values, dtype=float)))
        except Exception:
            med_vol_last = float('nan')
    # Final guard: fall back to current vol (or 1.0) if global median is unavailable
    if not np.isfinite(med_vol_last) or med_vol_last <= 0:
        med_vol_last = vol_now if np.isfinite(vol_now) and vol_now > 0 else 1.0

    sigs: List[Signal] = []
    thresholds: Dict[int, Dict[str, float]] = {}

    # Story 3.4: Compute per-step return cap from asset class
    _return_cap = RETURN_CAP_BY_CLASS.get(asset_type, RETURN_CAP_DEFAULT)

    # Story 3.3: Load previous signal state for two-day confirmation
    _prev_signal_state = load_signal_state(asset_key) if asset_key else {}

    # Story 4.5: Extract previous smoothed regime probabilities
    _prev_regime_probs_raw = _prev_signal_state.get("regime_probs")
    _prev_regime_probs = None
    if _prev_regime_probs_raw is not None:
        try:
            _prev_regime_probs = np.array(_prev_regime_probs_raw, dtype=float)
            if _prev_regime_probs.shape != (5,):
                _prev_regime_probs = None
        except (ValueError, TypeError):
            _prev_regime_probs = None

    # Regime-aware smoothing from HMM posterior uncertainty (replaces threshold-based)
    # Use regime persistence and uncertainty from posterior probabilities
    if regime_meta.get("method") == "hmm_posterior":
        # High persistence (diagonal transition prob) => low smoothing (trust signal)
        # Low persistence => high smoothing (reduce whipsaws)
        persistence = regime_meta.get("persistence", 0.5)
        alpha_edge = 0.30 + 0.25 * (1.0 - persistence)  # 0.30-0.55 range
        alpha_p = min(0.75, alpha_edge + 0.10)
    else:
        # Fallback for threshold-based regime
        alpha_edge = 0.40
        alpha_p = 0.50

    # v7.0: Removed _simulate_forward_paths (dual MC engine).
    # BMA MC now provides ALL samples (returns + volatility) at each horizon.
    # This eliminates the root cause of CRPS failure: two MC engines with
    # different volatility dynamics producing inconsistent p_up vs exp_ret.

    # v7.5: Multi-horizon BMA cache for calibration fast mode.
    # Call BMA MC once with H_max=max(horizons), extract all horizon samples.
    _bma_horizon_cache = {}  # populated on first iteration when _calibration_fast_mode=True

    # ==================================================================
    # REGIME DRIFT PRIOR (Story 1.2)
    # ==================================================================
    # Compute regime-conditional drift priors from historical data.
    # These provide Bayesian shrinkage targets for the Kalman posterior drift,
    # pulling extreme estimates toward regime-appropriate levels.
    #
    # The shrinkage operates via precision-weighted averaging:
    #   tau_kalman = 1 / P_t   (Kalman posterior precision)
    #   tau_prior  = 1 / se^2  (regime prior precision)
    #   mu_shrunk  = (tau_kalman * mu_kalman + tau_prior * mu_prior) / (tau_k + tau_p)
    # ==================================================================
    _regime_drift_prior = None
    _regime_drift_se = None
    try:
        _ret_s = feats.get("ret")
        _vol_s_rdp = feats.get("vol")
        if (isinstance(_ret_s, pd.Series) and isinstance(_vol_s_rdp, pd.Series)
                and len(_ret_s) >= 300):
            _rdp_result = _estimate_regime_drift_priors(_ret_s, _vol_s_rdp)
            if _rdp_result is not None:
                _regime_drift_prior = _rdp_result.get("current_drift_prior", 0.0)
                # Compute standard error of regime drift estimate
                _regime_drifts = _rdp_result.get("regime_drifts", {})
                _current_regime_hmm = _rdp_result.get("current_regime", "calm")
                # Use all regime drift values to estimate prior uncertainty
                _all_drifts = [v for v in _regime_drifts.values() if np.isfinite(v)]
                if len(_all_drifts) >= 2:
                    _regime_drift_se = float(np.std(_all_drifts, ddof=1))
                else:
                    _regime_drift_se = 0.001  # Weak prior when insufficient data
                if not np.isfinite(_regime_drift_prior):
                    _regime_drift_prior = None
                if _regime_drift_se is not None and (not np.isfinite(_regime_drift_se) or _regime_drift_se <= 0):
                    _regime_drift_se = 0.001
    except Exception:
        _regime_drift_prior = None
        _regime_drift_se = None

    for H in horizons:
        # ========================================================================
        # Unified Posterior Predictive Monte-Carlo Probability
        # ========================================================================
        # v7.0: ALL quantities (mH, sH, p_now, E_gain, E_loss) come from a
        # SINGLE set of BMA MC samples. No dual engine. No inconsistency.
        #
        # The BMA MC now uses run_unified_mc internally, which includes:
        # - GARCH(1,1) volatility dynamics
        # - Merton jump-diffusion
        # - Student-t innovations
        # - Full BMA mixture over models
        # ========================================================================

        # Extract drift posterior mean (μ̂_t) from Kalman filter or posterior drift
        mu_post_series = feats.get("mu_post", feats.get("mu_kf", pd.Series(dtype=float)))
        if isinstance(mu_post_series, pd.Series) and not mu_post_series.empty:
            mu_t_mc = float(mu_post_series.iloc[-1])
        else:
            mu_t_mc = 0.0
        if not np.isfinite(mu_t_mc):
            mu_t_mc = 0.0

        # ==================================================================
        # REGIME DRIFT PRIOR SHRINKAGE (Story 1.2)
        # ==================================================================
        # Apply Bayesian shrinkage toward regime-conditional drift prior.
        # This pulls extreme Kalman drift estimates toward regime-appropriate
        # levels, operating BEFORE momentum augmentation (Story 1.1).
        #
        # Precision-weighted update:
        #   tau_k = 1/P_t  (Kalman precision -- tight when filter is confident)
        #   tau_p = 1/se^2 (prior precision -- tight when regime drift is stable)
        #   mu_shrunk = (tau_k * mu_kalman + tau_p * mu_prior) / (tau_k + tau_p)
        #
        # When P_t is large (uncertain Kalman), shrinkage pulls strongly.
        # When P_t is small (confident Kalman), drift stays near Kalman estimate.
        # ==================================================================
        if _regime_drift_prior is not None and _regime_drift_se is not None:
            # Kalman posterior precision
            _P_t_for_shrink = feats.get("var_kf_smoothed", feats.get("var_kf", pd.Series(dtype=float)))
            if isinstance(_P_t_for_shrink, pd.Series) and not _P_t_for_shrink.empty:
                _P_t_val = float(_P_t_for_shrink.iloc[-1])
            else:
                _P_t_val = 1e-6
            if not np.isfinite(_P_t_val) or _P_t_val <= 0:
                _P_t_val = 1e-6

            _tau_kalman = 1.0 / max(_P_t_val, 1e-12)
            _tau_prior = 1.0 / max(_regime_drift_se ** 2, 1e-12)
            _tau_total = _tau_kalman + _tau_prior

            mu_t_mc = (_tau_kalman * mu_t_mc + _tau_prior * _regime_drift_prior) / _tau_total

            if not np.isfinite(mu_t_mc):
                mu_t_mc = 0.0

        # ==================================================================
        # MOMENTUM-AUGMENTED DRIFT (Story 1.1)
        # ==================================================================
        # Blend momentum-derived drift into mu_t_mc using horizon-dependent
        # weighting. Short horizons trust Kalman; long horizons trust momentum.
        #
        # w_mom(H) = min(1.0, H / MOM_CROSSOVER_HORIZON)
        # mu_augmented = (1 - w_mom) * mu_kalman + w_mom * mom_drift
        #
        # mom_drift = (momentum_score / 100) * vol_now * MOM_DRIFT_SCALE
        # ==================================================================
        _mu_t_mc_raw = mu_t_mc  # Preserve raw Kalman drift for diagnostics

        # Retrieve momentum features
        _mom21_s = feats.get("mom21")
        _mom63_s = feats.get("mom63")
        _mom126_s = feats.get("mom126")
        _mom252_s = feats.get("mom252")

        def _safe_last(series, default=0.0):
            if series is None:
                return default
            if isinstance(series, pd.Series) and not series.empty:
                v = float(series.iloc[-1])
                return v if np.isfinite(v) else default
            return default

        _m21 = _safe_last(_mom21_s)
        _m63 = _safe_last(_mom63_s)
        _m126 = _safe_last(_mom126_s)
        _m252 = _safe_last(_mom252_s)

        # Composite momentum: t-stat style, weighted by timeframe
        # These are already vol-normalized (cum_ret / realized_vol)
        _mom_weights = []
        _mom_values = []
        if abs(_m21) > 0 or _m21 == 0:
            _mom_weights.append(0.40)
            _mom_values.append(_m21)
        if abs(_m63) > 0 or _m63 == 0:
            _mom_weights.append(0.35)
            _mom_values.append(_m63)
        if abs(_m126) > 0 or _m126 == 0:
            _mom_weights.append(0.25)
            _mom_values.append(_m126)

        if _mom_values:
            _tw = sum(_mom_weights)
            _composite_mom = sum(w * v for w, v in zip(_mom_weights, _mom_values)) / _tw
        else:
            _composite_mom = 0.0

        # Determine observation count for safety gate
        _vol_s = feats.get("vol", pd.Series(dtype=float))
        _n_obs = len(_vol_s) if isinstance(_vol_s, pd.Series) else 0
        _vol_now_mom = float(_vol_s.iloc[-1]) if (isinstance(_vol_s, pd.Series) and not _vol_s.empty) else 0.01
        if not np.isfinite(_vol_now_mom) or _vol_now_mom <= 0:
            _vol_now_mom = 0.01

        # Convert composite momentum (t-stat, typically [-3, +3]) to daily return units
        # mom_drift = composite_mom * vol_now * MOM_DRIFT_SCALE
        # Clamp composite to [-3, +3] to prevent extreme outliers
        _composite_mom_clamped = float(np.clip(_composite_mom, -3.0, 3.0))
        _mom_drift = _composite_mom_clamped * _vol_now_mom * MOM_DRIFT_SCALE

        # Horizon-dependent blending weight
        _w_mom = min(1.0, H / MOM_CROSSOVER_HORIZON)

        # Safety gates: disable momentum augmentation when data is insufficient
        # or momentum is NaN/zero (no directional information)
        _mom_enabled = (
            _n_obs >= MOM_MIN_OBSERVATIONS
            and np.isfinite(_mom_drift)
            and abs(_composite_mom_clamped) > 1e-9
        )

        if _mom_enabled:
            mu_t_mc = (1.0 - _w_mom) * _mu_t_mc_raw + _w_mom * _mom_drift
        # else: mu_t_mc stays as raw Kalman drift

        # Extract drift posterior variance (P_t) from Kalman filter
        var_kf_series_prob = feats.get("var_kf_smoothed", feats.get("var_kf", pd.Series(dtype=float)))
        if isinstance(var_kf_series_prob, pd.Series) and not var_kf_series_prob.empty:
            P_t_mc = float(var_kf_series_prob.iloc[-1])
        else:
            P_t_mc = 0.0
        if not np.isfinite(P_t_mc) or P_t_mc < 0:
            P_t_mc = 0.0

        # ========================================================================
        # REGIME-FIRST PARAMETER ROUTING (STEP 1 & 2)
        # ========================================================================
        current_regime_idx = map_regime_label_to_index(reg, regime_meta)
        km_mc = feats.get("kalman_metadata", {}) or {}
        tuned_params_full = {
            'q': km_mc.get("process_noise_var", 1e-6),
            'phi': km_mc.get("phi_used") or km_mc.get("kalman_phi") or 0.95,
            'nu': km_mc.get("kalman_nu"),
            'c': km_mc.get("kalman_c_optimal", 1.0),
            'noise_model': km_mc.get("kalman_noise_model", "gaussian"),
            'regime': km_mc.get("regime_params", {}),
            'has_regime_params': km_mc.get("has_regime_params", False),
        }
        theta = _select_regime_params(tuned_params_full, current_regime_idx)
        q_mc = float(theta.get("q", 1e-6))
        phi_mc = float(theta.get("phi", 0.95))
        nu_mc = theta.get("nu")
        c_mc = float(theta.get("c", 1.0))
        collapse_warning = theta.get("collapse_warning", False)
        regime_source = theta.get("source", "unknown")
        regime_used = theta.get("regime_used", current_regime_idx)
        noise_model_mc = km_mc.get("kalman_noise_model", "gaussian")
        if nu_mc is not None and (not np.isfinite(nu_mc) or nu_mc <= 2.0):
            nu_mc = None
        is_student_t_mc = noise_model_mc and noise_model_mc.startswith('phi_student_t_nu_')
        if not is_student_t_mc or nu_mc is None:
            noise_model_mc = "gaussian"

        # ========================================================================
        # VOLATILITY GEOMETRY: sigma2_step is the PRIMITIVE
        # ========================================================================
        vol_series_mc = feats.get("vol", pd.Series(dtype=float))
        if isinstance(vol_series_mc, pd.Series) and not vol_series_mc.empty:
            sigma_now = float(vol_series_mc.iloc[-1])
        else:
            sigma_now = 0.01
        if not np.isfinite(sigma_now) or sigma_now <= 0:
            sigma_now = 0.01
        sigma2_step_mc = float(sigma_now ** 2)

        # ========================================================================
        # BUILD REGIME PARAMS AND CALL BMA MC (v7.0: SINGLE SOURCE OF TRUTH)
        # ========================================================================
        regime_params = {}
        cached_regime_params = km_mc.get("regime_params", {})
        for regime_idx in range(5):
            regime_theta = _select_regime_params(tuned_params_full, regime_idx)
            regime_params[regime_idx] = {
                "phi": regime_theta.get("phi", phi_mc),
                "q": regime_theta.get("q", q_mc),
                "nu": regime_theta.get("nu", nu_mc),
                "c": regime_theta.get("c", c_mc),
                "fallback": regime_theta.get("fallback", True),
            }

        # v7.0: Single unified BMA MC call — returns BOTH return and vol samples
        # v7.5: In _calibration_fast_mode, call once with H_max and cache all horizons
        # Story 1.4: Compute dual-frequency drift parameters
        _phi_slow_val = math.exp(-1.0 / MOMENTUM_HALF_LIFE_DAYS) if MOMENTUM_HALF_LIFE_DAYS > 0 else 0.0
        _mu_slow_0_val = mu_t_mc * MOM_SLOW_FRAC
        # Adjust fast component: mu_t_mc for the kernel is the fast part only
        mu_t_mc = mu_t_mc * (1.0 - MOM_SLOW_FRAC)
        if _calibration_fast_mode and _bma_horizon_cache:
            # Reuse cached BMA samples from first iteration
            r_samples = _bma_horizon_cache["samples"].get(H, np.array([0.0]))
            vol_samples_bma = _bma_horizon_cache["vol_samples"].get(H, np.array([0.0]))
            regime_probs = _bma_horizon_cache["regime_probs"]
            bma_meta = _bma_horizon_cache["bma_meta"]
        else:
            _bma_H = max(horizons) if _calibration_fast_mode else H
            _bma_hz_extract = horizons if _calibration_fast_mode else None
            r_samples, vol_samples_bma, regime_probs, bma_meta = bayesian_model_average_mc(
                feats=feats,
                regime_params=regime_params,
                mu_t=mu_t_mc,
                P_t=P_t_mc,
                sigma2_step=sigma2_step_mc,
                H=_bma_H,
                n_paths=n_mc_paths,
                seed=None,
                tuned_params=tuned_params,
                asset_symbol=asset_key,
                horizons_extract=_bma_hz_extract,
                # Story 1.4: Dual-frequency drift
                phi_slow=_phi_slow_val,
                mu_slow_0=_mu_slow_0_val,
                # Story 3.4: Asset-class-aware return cap
                return_cap=_return_cap,
                # Story 4.5: Previous smoothed regime probs
                prev_regime_probs=_prev_regime_probs,
            )
            if _calibration_fast_mode:
                _bma_horizon_cache = {
                    "samples": bma_meta.get("horizon_samples", {}),
                    "vol_samples": bma_meta.get("horizon_vol_samples", {}),
                    "regime_probs": regime_probs,
                    "bma_meta": bma_meta,
                }
                # Use current H's samples
                r_samples = _bma_horizon_cache["samples"].get(H, r_samples)
                vol_samples_bma = _bma_horizon_cache["vol_samples"].get(H, vol_samples_bma)
        r = np.asarray(r_samples, dtype=float)

        # Story 3.6: MC path diagnostics
        # Wrap 1-D r_samples as (1, n_paths) for diagnose_mc_paths
        _diag_arr = r.reshape(1, -1) if r.ndim == 1 else r
        _mc_diag = diagnose_mc_paths(
            _diag_arr, H=1, asset_name=asset_key or "",
        )

        # v7.0: Use BMA samples for ALL quantities (unified MC)
        # Story 3.6: Exclude NaN paths
        sim_H = r[np.isfinite(r)]
        if sim_H.size == 0:
            sim_H = np.zeros(3000, dtype=float)
        vol_H = np.asarray(vol_samples_bma, dtype=float)
        vol_H = vol_H[np.isfinite(vol_H)]
        if vol_H.size == 0:
            vol_H = np.zeros(3000, dtype=float)

        # ========================================================================
        # Compute moments from unified BMA samples (v7.0)
        # ========================================================================
        mH = float(np.median(sim_H))
        vH = float(np.var(sim_H, ddof=1)) if sim_H.size > 1 else 0.0
        sH = float(math.sqrt(max(vH, 1e-12)))
        z_stat = float(mH / sH) if sH > 0 else 0.0

        # p_now, E_gain, E_loss from SAME samples
        p_now = float(np.mean(r > 0.0))

        # === SIGNAL CALIBRATION: p_up recalibration (Pass 2) ===
        _sig_cal = _load_signals_calibration(tuned_params)
        p_now = _apply_p_up_calibration(p_now, _sig_cal, H, vol_regime=vol_reg_now)

        # ================================================================
        # ISOTONIC CALIBRATION — REAL PROBABILITY IMPROVEMENT (March 2026)
        # ================================================================
        # Apply the isotonic transport map learned during tuning to correct
        # systematic biases in P(up). The map was fitted on PIT values which
        # measure how well the model's CDF matches realized outcomes.
        #
        # Key insight: if the model systematically overestimates P(up),
        # the isotonic map learns this bias and corrects it. This directly
        # improves EU = p × E[gain] - (1-p) × E[loss] → better sizing.
        #
        # Uses simple linear interpolation on stored knots (fast, no scipy).
        # ================================================================
        _isotonic_applied = False
        _cal_params = feats.get("calibration_params", {})
        _iso_x = _cal_params.get("isotonic_x_knots")
        _iso_y = _cal_params.get("isotonic_y_knots")
        if _iso_x is not None and _iso_y is not None:
            try:
                _iso_x_arr = np.asarray(_iso_x, dtype=np.float64)
                _iso_y_arr = np.asarray(_iso_y, dtype=np.float64)
                if len(_iso_x_arr) >= 2 and len(_iso_y_arr) >= 2:
                    # Linear interpolation of p_now through the transport map
                    p_calibrated = float(np.interp(p_now, _iso_x_arr, _iso_y_arr))
                    p_calibrated = float(np.clip(p_calibrated, 0.01, 0.99))
                    # Safety: don't allow isotonic to flip direction
                    if (p_now > 0.5 and p_calibrated > 0.5) or \
                       (p_now < 0.5 and p_calibrated < 0.5) or \
                       abs(p_now - 0.5) < 0.05:
                        p_now = p_calibrated
                        _isotonic_applied = True
            except Exception:
                pass  # Isotonic calibration is best-effort

        gains = r[r > 0.0]
        losses = -r[r < 0.0]

        E_gain = float(np.mean(gains)) if gains.size > 0 else 0.0
        E_loss_empirical = float(np.mean(losses)) if losses.size > 0 else 0.0

        # ====================================================================
        # EVT-CORRECTED EXPECTED LOSS (Expert Panel Solution 2)
        # ====================================================================
        # The Pickands–Balkema–de Haan theorem provides theoretical foundation:
        # exceedances over high threshold u converge to GPD distribution.
        #
        # CTE = E[Loss | Loss > u] = u + σ/(1-ξ)  for ξ < 1
        #
        # This replaces the naive empirical mean with principled extrapolation
        # that captures extreme tail behavior beyond observed MC samples.
        #
        # Key properties:
        #   - EVT E[loss] ≥ empirical E[loss] (always more conservative)
        #   - Heavy-tailed assets (ξ > 0.2) get larger loss estimates
        #   - Light-tailed assets (ξ ≈ 0) get minimal adjustment
        #   - Fallback to 1.5× empirical if GPD fitting fails
        # ====================================================================
        
        # Initialize EVT diagnostics
        evt_expected_loss = E_loss_empirical
        evt_gpd_result = None
        evt_enabled = False
        evt_xi = None
        evt_sigma = None
        evt_threshold = None
        evt_n_exceedances = 0
        evt_fit_method = None
        evt_consistency = None
        
        if EVT_AVAILABLE and losses.size >= EVT_MIN_EXCEEDANCES:
            try:
                # Compute EVT-corrected expected loss
                evt_loss, emp_loss, gpd_result = compute_evt_expected_loss(
                    r_samples=r,
                    threshold_percentile=EVT_THRESHOLD_PERCENTILE_DEFAULT,
                    fallback_multiplier=EVT_FALLBACK_MULTIPLIER
                )
                
                evt_expected_loss = evt_loss
                evt_gpd_result = gpd_result
                evt_enabled = True
                evt_xi = gpd_result.xi
                evt_sigma = gpd_result.sigma
                evt_threshold = gpd_result.threshold
                evt_n_exceedances = gpd_result.n_exceedances
                evt_fit_method = gpd_result.method
                
                # Check consistency with Student-t ν (if available)
                if nu_mc is not None and gpd_result.fit_success:
                    evt_consistency = check_student_t_consistency(nu_mc, gpd_result.xi)
                    
            except Exception as evt_err:
                # EVT failed - fall back to empirical × multiplier
                evt_expected_loss = E_loss_empirical * EVT_FALLBACK_MULTIPLIER
                evt_fit_method = 'exception_fallback'
        elif losses.size > 0:
            # Insufficient data for EVT - use conservative fallback
            evt_expected_loss = E_loss_empirical * EVT_FALLBACK_MULTIPLIER
            evt_fit_method = 'insufficient_data_fallback'
        
        # Use EVT-corrected loss for position sizing
        # Story 5.2: Cap EVT inflation to prevent E_loss blow-up
        if E_loss_empirical > 0:
            E_loss = min(evt_expected_loss, EVT_MAX_INFLATION * E_loss_empirical)
        else:
            E_loss = evt_expected_loss

        # Story 5.2: Asymmetric EU (legacy, for diagnostics)
        EU_asymmetric = p_now * E_gain - (1.0 - p_now) * E_loss

        # Story 5.2: Balanced EU (symmetric tail treatment for position sizing)
        _xi_for_gain = float(evt_xi) if evt_xi is not None and np.isfinite(evt_xi) else 0.0
        _gain_correction = min(1.0 + abs(_xi_for_gain) * EVT_GAIN_FACTOR, EVT_MAX_INFLATION)
        E_gain_evt = E_gain * _gain_correction
        EU_balanced = p_now * E_gain_evt - (1.0 - p_now) * E_loss

        # Use balanced EU for trading decisions
        EU = EU_balanced

        epsilon_eu = 1e-12
        max_position_size = 1.0

        if EU > 0.0 and E_loss > 0.0:
            eu_position_size = EU / max(E_loss, epsilon_eu)
        else:
            eu_position_size = 0.0

        # clip to risk limits
        eu_position_size = float(np.clip(eu_position_size, 0.0, max_position_size))

        # Story 5.6: Kelly Criterion sizing
        KELLY_CAP = 0.25
        if E_loss > epsilon_eu and E_gain > 0:
            _odds = E_gain / E_loss
            kelly_full = (p_now * _odds - (1.0 - p_now)) / _odds
            kelly_full = float(np.clip(kelly_full, 0.0, KELLY_CAP))
            kelly_half = kelly_full / 2.0
        else:
            kelly_full = 0.0
            kelly_half = 0.0

        # Expected Utility metrics for logging/Signal
        expected_utility = EU
        expected_gain = E_gain
        expected_loss = E_loss
        expected_loss_empirical = E_loss_empirical  # Keep for comparison
        gain_loss_ratio = E_gain / max(E_loss, epsilon_eu) if E_loss > epsilon_eu else (
            100.0 if E_gain > 0 else 1.0
        )

        # For diagnostics: compute drift uncertainty propagated to horizon
        # (kept for Signal dataclass but NOT used for trading)
        if phi_mc is not None and np.isfinite(phi_mc) and abs(phi_mc) < 0.999:
            phi2_diag = phi_mc ** 2
            if abs(1.0 - phi2_diag) > 1e-10:
                drift_var_factor = (1.0 - phi2_diag ** H) / (1.0 - phi2_diag)
            else:
                drift_var_factor = float(H)
        else:
            drift_var_factor = float(H)
        drift_uncertainty_H = drift_var_factor * P_t_mc

        # Diagnostic probabilities (NOT used for trading, kept for analysis only)
        # These are stored in Signal for monitoring but p_now is the only trading probability
        p_empirical = float(np.mean(sim_H > 0.0))  # Raw empirical from simulation
        predictive_var_diag = vH + drift_uncertainty_H
        predictive_std_diag = float(math.sqrt(max(predictive_var_diag, 1e-12)))
        z_predictive_diag = float(mH / predictive_std_diag) if predictive_std_diag > 0 else 0.0
        # Check for Student-t model (phi_student_t_nu_* naming)
        is_student_t_diag = noise_model_mc and noise_model_mc.startswith('phi_student_t_nu_')
        if is_student_t_diag and nu_mc is not None:
            p_analytical = float(student_t.cdf(z_predictive_diag, df=float(nu_mc)))
        else:
            p_analytical = float(norm.cdf(z_predictive_diag))
        p_analytical = float(np.clip(p_analytical, 0.001, 0.999))
        p_posterior_predictive = p_analytical  # Alias for backward compatibility

        # Expected log return and CI from simulation (percentile CI)
        # Define quantile bounds early for use in volatility CI as well
        q = float(np.clip(ci, 1e-6, 0.999999))
        lo_q = (1.0 - q) / 2.0
        hi_q = 1.0 - lo_q

        # Stochastic volatility statistics (Level-7: full posterior uncertainty)
        vol_mean = float(np.mean(vol_H)) if vol_H.size > 0 else 0.0
        try:
            vol_ci_low = float(np.quantile(vol_H, lo_q))
            vol_ci_high = float(np.quantile(vol_H, hi_q))
        except Exception:
            vol_std = float(np.std(vol_H)) if vol_H.size > 1 else 0.0
            vol_ci_low = max(0.0, vol_mean - vol_std)
            vol_ci_high = vol_mean + vol_std
        # Story 3.3: Load previous p_up from persisted state for real 2-day confirmation.
        # On first run (no state), fall back to p_now to preserve pre-3.3 behavior.
        # On subsequent runs, p_prev comes from the previous day's saved state.
        _h_key = str(H)
        _prev_h = _prev_signal_state.get(_h_key, {})
        p_prev = float(_prev_h.get("p_up", p_now))  # first run: p_prev=p_now (backward compat)
        p_s_prev = p_prev
        p_s_now = alpha_p * p_now + (1.0 - alpha_p) * p_prev

        # Base/composite edge built off z_stat to keep consistency
        base_now = z_stat
        base_prev = z_stat  # lacking prev sim, reuse now as stable default
        edge_prev = composite_edge(base_prev, trend_prev, moms_prev, vol_reg_prev, z5_prev)
        edge_now = composite_edge(base_now, trend_now, moms_now, vol_reg_now, z5_now)

        # Expected return CI from simulation (percentile CI) - lo_q, hi_q already defined above
        try:
            ci_low = float(np.quantile(sim_H, lo_q))
            ci_high = float(np.quantile(sim_H, hi_q))
        except Exception:
            ci_low = mH - 1.0 * sH
            ci_high = mH + 1.0 * sH

        # ========================================================================
        # DISPLAYED EXPECTED RETURN: MC Median (v7.0)
        # ========================================================================
        # Use the median of the MC posterior predictive (mH) directly as
        # the displayed expected return. Median is the robust location
        # estimator for heavy-tailed distributions.
        #
        # EU (Expected Utility) is used ONLY for:
        #   - BUY/SELL/HOLD/EXIT label logic
        #   - Position sizing (eu_position_size)
        # EU must NOT override the displayed return direction, because
        # EVT-inflated E_loss can make EU negative even when the model
        # predicts positive returns — creating a systematic negative bias.
        # ========================================================================
        mu_H = mH  # MC median: the model's robust prediction

        # === SIGNAL CALIBRATION: EMOS distributional correction (Pass 2) ===
        # v3.0: EMOS (Gneiting 2005) — unified 4-param affine correction
        # optimized via CRPS. Replaces both mag_scale and bias. Also
        # corrects sigma for properly calibrated uncertainty.
        # v2.0 fallback: mag_scale + bias.
        mu_H, sig_H = _apply_emos_correction(mu_H, sH, _sig_cal, H, vol_regime=vol_reg_now)

        # ========================================================================
        # SAFETY: Cap sig_H to prevent absurd CI bounds (March 2026)
        # ========================================================================
        # For extreme-vol assets (e.g., ABTC daily σ≈50%), uncapped sig_H grows
        # as σ_daily × √H, reaching 800%+ at H=252. This produces CI displays
        # like [-1,832,151%, +1,831,…%]. Cap sig_H using asset-type-aware
        # annual volatility limits with √H scaling.
        # ========================================================================
        _sig_h_cap = _compute_sig_h_cap(H, asset_type)
        sig_H = min(sig_H, _sig_h_cap)

        # ========================================================================
        # SAFETY: Clamp mu_H to prevent exp() overflow and absurd forecasts
        # ========================================================================
        # Principled bound: the median forecast should not exceed ±4σ of the
        # model's own calibrated uncertainty (sig_H).  A √H-scaled floor
        # prevents over-tightening for very low-vol assets.
        #
        # Absolute bounds (safety valves):
        #   Upside:  ln(5) ≈ 1.61  →  exp(1.61)-1 ≈ 400%  (median can't predict 5×)
        #   Downside: -4.6          →  exp(-4.6)-1 ≈ -99%  (near-total loss)
        # ========================================================================
        _vol_cap = 4.0 * sig_H                             # 4σ from calibrated uncertainty
        _floor_cap = 0.03 * math.sqrt(max(H, 1))           # floor for very low-vol assets
        _mu_H_cap = max(_vol_cap, _floor_cap)
        mu_H = float(np.clip(mu_H, max(-_mu_H_cap, -4.6), min(_mu_H_cap, 1.61)))

        # ========================================================================
        # EXPECTED UTILITY POSITION SIZING (REPLACES KELLY/MEAN-BASED SIZING)
        # ========================================================================
        # All sizing is now derived from the full posterior predictive distribution
        # (r_samples from BMA), NOT from point estimates.
        #
        # Design Principle:
        #   - Inference produces distributions (r_samples)
        #   - Decisions must consume distributions, not point estimates
        #   - Kelly formula (f_star = mu_H / denom) is PROHIBITED
        #
        # Expected Utility Model:
        #   EU = p × E[gain] - (1-p) × E[loss]
        #   size = EU / max(E[loss], ε)
        #
        # Key Properties:
        #   - Two assets with identical p can have different sizes
        #   - Fat downside tails → higher E[loss] → smaller size
        #   - Strong upside asymmetry → higher E[gain] → larger size
        #   - EU ≤ 0 → HOLD (no position)
        # ========================================================================

        # ====================================================================
        # CALIBRATED TRUST AUTHORITY — SINGLE POINT OF TRUST DECISION
        # ====================================================================
        # ARCHITECTURAL LAW: Trust = Calibration Authority − Regime Penalty
        #
        # This replaces the old hard-coded threshold logic with principled
        # additive decomposition. Calibration speaks first, regimes discount.
        #
        # SCORING (Counter-Proposal v2):
        #   Authority discipline:           98/100
        #   Mathematical transparency:      97/100
        #   Audit traceability:             97/100
        # ====================================================================

        kalman_metadata = feats.get("kalman_metadata", {})
        pit_pvalue = kalman_metadata.get("pit_ks_pvalue")

        # Try to use calibrated trust from tuned params (preferred path)
        calibrated_trust_data = feats.get("calibrated_trust")

        if CALIBRATED_TRUST_AVAILABLE and calibrated_trust_data is not None:
            # Load pre-computed calibrated trust from tuning
            try:
                trust = CalibratedTrust.from_dict(calibrated_trust_data)
                drift_weight = compute_drift_weight(trust, min_weight=0.1, max_weight=1.0)
                
                # v2.0: Check if elite fragility penalty was included in cached trust
                elite_penalty_from_cache = trust.audit_decomposition.get('elite_fragility_penalty', 0.0)

                # Store for diagnostics
                feats["trust_audit"] = {
                    "calibration_trust": trust.calibration_trust,
                    "regime_penalty": trust.regime_penalty,
                    "effective_trust": trust.effective_trust,
                    "drift_weight": drift_weight,
                    "source": "cached_trust",
                    # v2.0: Elite fragility impact on signals
                    "elite_fragility_penalty": elite_penalty_from_cache,
                    "elite_diagnostics_used": elite_penalty_from_cache > 0,
                }
            except Exception as e:
                # Fallback to computing trust on-the-fly
                calibrated_trust_data = None

        if not CALIBRATED_TRUST_AVAILABLE or calibrated_trust_data is None:
            # Compute calibrated trust on-the-fly from available PIT data
            drift_weight = 1.0  # Default: trust EU sizing fully

            if CALIBRATED_TRUST_AVAILABLE and pit_pvalue is not None:
                # Build PIT samples if we have recalibration data
                recal_data = feats.get("recalibration")

                if recal_data is not None:
                    # Use stored calibrated PIT
                    calibrated_pit = np.array(recal_data.get("calibrated_pit", []))
                    if len(calibrated_pit) > 0:
                        # Use SOFT regime probabilities from BMA (not hard assignment)
                        # This avoids penalty cliffs at regime boundaries
                        # regime_probs is now a Dict[int, float] from bayesian_model_average_mc
                        soft_regime_probs_for_trust = regime_probs if isinstance(regime_probs, dict) else {1: 1.0}

                        # Extract elite diagnostics for fragility penalty (v2.0 - February 2026)
                        # This directly affects BUY/SELL/HOLD/EXIT signal strength
                        elite_diag_for_trust = None
                        if kalman_metadata:
                            fragility_idx = kalman_metadata.get('elite_fragility_index')
                            if fragility_idx is not None:
                                elite_diag_for_trust = {
                                    'fragility_index': fragility_idx,
                                    'is_ridge_optimum': kalman_metadata.get('elite_is_ridge', False),
                                    'basin_score': kalman_metadata.get('elite_basin_score', 1.0),
                                    'drift_ratio': kalman_metadata.get('elite_drift_ratio', 0.0),
                                }

                        try:
                            trust = compute_calibrated_trust(
                                raw_pit_values=calibrated_pit,
                                regime_probs=soft_regime_probs_for_trust,
                                config=TrustConfig(),
                                elite_diagnostics=elite_diag_for_trust,  # v2.0: affects signals
                            )
                            drift_weight = compute_drift_weight(trust, min_weight=0.1, max_weight=1.0)

                            feats["trust_audit"] = {
                                "calibration_trust": trust.calibration_trust,
                                "regime_penalty": trust.regime_penalty,
                                "effective_trust": trust.effective_trust,
                                "drift_weight": drift_weight,
                                "source": "computed_on_fly_soft_regime",
                                "soft_regime_probs": soft_regime_probs_for_trust,
                                # v2.0: Elite fragility impact on signals
                                "elite_fragility_penalty": trust.audit_decomposition.get('elite_fragility_penalty', 0.0),
                                "elite_diagnostics_used": elite_diag_for_trust is not None,
                            }
                        except Exception:
                            pass  # Fall through to legacy logic

            # Legacy fallback: hard threshold (preserved for backward compatibility)
            if "trust_audit" not in feats:
                if pit_pvalue is not None and np.isfinite(pit_pvalue) and pit_pvalue < 0.05:
                    # Calibration warning: model forecasts not well-calibrated
                    drift_weight = 0.3

                feats["trust_audit"] = {
                    "calibration_trust": pit_pvalue if pit_pvalue is not None else 0.5,
                    "regime_penalty": 0.0,
                    "effective_trust": drift_weight,
                    "drift_weight": drift_weight,
                    "source": "legacy_threshold",
                }

        # === FINAL POSITION STRENGTH ===
        # Story 5.3: Blend EU-based sizing with forecast-magnitude sizing
        # EU size (risk-adjusted): EU / max(E_loss, eps), clipped to [0, 1]
        _size_eu = float(np.clip(eu_position_size, 0.0, 1.0))
        # Magnitude size (conviction-based): |mu_H| / sig_H info ratio
        _size_mag = abs(mu_H) / (sig_H + 1e-6) if sig_H > 0 else 0.0
        _size_mag = float(np.clip(_size_mag, 0.0, 1.0))
        blended_position_size = SIZE_EU_WEIGHT * _size_eu + SIZE_MAG_WEIGHT * _size_mag
        blended_position_size = float(np.clip(blended_position_size, 0.0, 1.0))
        # Story 5.6: Kelly half provides a floor for genuine signals
        blended_position_size = max(blended_position_size, kelly_half)
        blended_position_size = float(np.clip(blended_position_size, 0.0, 1.0))
        pos_strength = drift_weight * blended_position_size

        # Logging/diagnostics: p, E_gain, E_loss, EU, pos_strength
        # (These are stored in Signal dataclass for analysis)

        # Level-7 Modularization: Use helper function for dynamic thresholds
        # Enriched regime_meta with vol_regime for fallback path
        regime_meta_enriched = dict(regime_meta)
        regime_meta_enriched["vol_regime"] = vol_reg_now

        threshold_result = compute_dynamic_thresholds(
            skew=skew_now,
            regime_meta=regime_meta_enriched,
            sig_H=sig_H,
            med_vol_last=med_vol_last,
            H=H
        )

        buy_thr = threshold_result["buy_thr"]
        sell_thr = threshold_result["sell_thr"]
        U = threshold_result["uncertainty"]

        # === SIGNAL CALIBRATION: per-asset label thresholds (Pass 2) ===
        # Override dynamic thresholds with calibrated per-asset thresholds
        # if available from two-pass tuning walk-forward optimization.
        # v2.0: per-horizon thresholds with fallback to global.
        _cal_thresholds = _get_calibrated_label_thresholds(_sig_cal, H=H)
        if _cal_thresholds is not None:
            buy_thr, sell_thr = _cal_thresholds

        # Story 5.1: Adaptive edge floor based on asset volatility
        _adaptive_edge_floor = compute_adaptive_edge_floor(vol_now, H)

        thresholds[int(H)] = {
            "buy_thr": float(buy_thr),
            "sell_thr": float(sell_thr),
            "uncertainty": float(U),
            "edge_floor": float(_adaptive_edge_floor)
        }

        # Level-7 Modularization: Use helper function for confirmation logic
        label = apply_confirmation_logic(
            p_smoothed_now=p_s_now,
            p_smoothed_prev=p_s_prev,
            p_raw=p_now,
            pos_strength=pos_strength,
            buy_thr=buy_thr,
            sell_thr=sell_thr,
            edge=edge_now,
            edge_floor=_adaptive_edge_floor
        )

        # CI bounds for expected log return
        # Story 3.2: Use quantile-based CIs from MC samples (primary).
        # Parametric CI (mu +/- z*sig) is ONLY used when n_samples < 100.
        # Quantile CIs correctly capture asymmetry and heavy tails of BMA distribution.
        _n_sim = len(sim_H) if sim_H is not None else 0
        if _n_sim >= QUANTILE_CI_MIN_SAMPLES:
            # Primary: empirical quantiles from MC posterior predictive
            _alpha_ci = (1.0 - ci) / 2.0  # ci=0.68 -> alpha=0.16
            ci_low = float(np.quantile(sim_H, _alpha_ci))
            ci_high = float(np.quantile(sim_H, 1.0 - _alpha_ci))
            # 90% CI for risk assessment
            ci_low_90 = float(np.quantile(sim_H, 0.05))
            ci_high_90 = float(np.quantile(sim_H, 0.95))
        else:
            # Fallback: parametric CI (only when samples < 100)
            ci_low = float(mu_H - z_star * sig_H)
            ci_high = float(mu_H + z_star * sig_H)
            ci_low_90 = float(mu_H - 1.645 * sig_H)
            ci_high_90 = float(mu_H + 1.645 * sig_H)

        # Clamp CI to physical limits (March 2026)
        # Can't predict worse than -99% loss or better than +400% gain
        ci_low = max(ci_low, _CI_LOG_FLOOR)
        ci_high = min(ci_high, _CI_LOG_CAP)
        ci_low_90 = max(ci_low_90, _CI_LOG_FLOOR)
        ci_high_90 = min(ci_high_90, _CI_LOG_CAP)
        # Ensure CI ordering after clamping
        if ci_low > ci_high:
            ci_low, ci_high = ci_high, ci_low
        if ci_low_90 > ci_high_90:
            ci_low_90, ci_high_90 = ci_high_90, ci_low_90

        # Convert expected log‑return to PLN profit for a 1,000,000 PLN notional
        exp_mult = float(np.exp(mu_H))
        ci_low_mult = float(np.exp(ci_low))
        ci_high_mult = float(np.exp(ci_high))

        # Raw (unsmoothed) profit values — clamped to physical limits
        # Can't lose more than 100% (price ≥ 0) or gain more than 400%
        raw_profit_pln = float(np.clip(NOTIONAL_PLN * (exp_mult - 1.0),
                                       -NOTIONAL_PLN, 4.0 * NOTIONAL_PLN))
        raw_profit_ci_low_pln = float(np.clip(NOTIONAL_PLN * (ci_low_mult - 1.0),
                                              -NOTIONAL_PLN, 4.0 * NOTIONAL_PLN))
        raw_profit_ci_high_pln = float(np.clip(NOTIONAL_PLN * (ci_high_mult - 1.0),
                                               -NOTIONAL_PLN, 4.0 * NOTIONAL_PLN))

        # ========================================================================
        # UPGRADE #3: Display Price Inertia (Presentation-Only)
        # ========================================================================
        # Apply smoothing to displayed profit to reduce day-to-day jitter.
        # Formula: display_price = 0.7 * prev_display_price + 0.3 * new_predicted_price
        #
        # IMPORTANT: This does NOT affect trading decisions, EU, or regimes.
        # It only prevents "why did this jump?" moments for users.
        # ========================================================================
        if asset_key is not None and len(asset_key) > 0:
            profit_pln = _smooth_display_price(asset_key, H, raw_profit_pln)
            profit_ci_low_pln = _smooth_display_price(f"{asset_key}_lo", H, raw_profit_ci_low_pln)
            profit_ci_high_pln = _smooth_display_price(f"{asset_key}_hi", H, raw_profit_ci_high_pln)
        else:
            profit_pln = raw_profit_pln
            profit_ci_low_pln = raw_profit_ci_low_pln
            profit_ci_high_pln = raw_profit_ci_high_pln

        # ========================================================================
        # DUAL-SIDED TREND EXHAUSTION (UE↑ / UE↓) - MULTI-TIMEFRAME
        # ========================================================================
        # Compute directional exhaustion using weighted multi-timeframe EMA
        # deviation with Student-t fat-tail corrections.
        #
        # Output: 0-100% scale indicating how far price deviates from equilibrium
        # - ue_up: Price above weighted EMA equilibrium (higher = more extended)
        # - ue_down: Price below weighted EMA equilibrium (higher = more extended)
        # - Mutual exclusivity: only one can be non-zero
        # - Same value for all horizons (price-based, not model-based)
        # ========================================================================

        # Compute exhaustion from price features (same for all horizons)
        exh_result = compute_directional_exhaustion_from_features(feats)

        ue_up = exh_result["ue_up"]
        ue_down = exh_result["ue_down"]

        # ========================================================================
        # EXHAUSTION-BASED RISK MODULATION (SOFT ONLY)
        # Story 5.5: Direction-aware exhaustion modulation
        # ========================================================================
        # Long signals (mu_H > 0):
        #   ue_up (overbought)  -> reduce position (caution)
        #   ue_down (oversold)  -> increase position (mean-reversion opportunity)
        # Short signals (mu_H <= 0):
        #   ue_up (overbought)  -> increase position (mean-reversion opportunity)
        #   ue_down (oversold)  -> reduce position (caution)
        # Increase factor (0.3) < decrease factor (0.5) for prudence.
        # Position always capped at 1.0.
        # ========================================================================
        _exh_reduce = 0.5   # Reduction factor (caution)
        _exh_boost = 0.3    # Boost factor (mean-reversion, conservative)
        if mu_H > 0:  # Long signal
            if ue_up > 0 and pos_strength > 0:
                pos_strength *= (1.0 - _exh_reduce * ue_up)
            if ue_down > 0 and pos_strength > 0:
                pos_strength *= (1.0 + _exh_boost * ue_down)
        else:  # Short / neutral signal
            if ue_up > 0 and pos_strength > 0:
                pos_strength *= (1.0 + _exh_boost * ue_up)
            if ue_down > 0 and pos_strength > 0:
                pos_strength *= (1.0 - _exh_reduce * ue_down)
        pos_strength = min(pos_strength, 1.0)

        # =====================================================================
        # RISK TEMPERATURE MODULATION (Expert Panel Solution 1 + 4)
        # =====================================================================
        # Scale position strength based on cross-asset stress indicators.
        # This is the final modulation layer BEFORE position output.
        #
        # DESIGN: pos_strength_final = pos_strength_base × scale_factor(temp)
        #
        # Stress categories:
        #   - FX (40%): AUDJPY, USDJPY, CHF — risk-on/off proxy
        #   - Futures (30%): ES/NQ momentum — equity sentiment
        #   - Rates (20%): TLT volatility — macro stress
        #   - Commodities (10%): Copper, gold/copper — growth fear
        #
        # Scaling:
        #   - temp = 0.0 → scale ≈ 0.95 (near-full exposure)
        #   - temp = 1.0 → scale = 0.50 (half exposure)
        #   - temp = 2.0 → scale ≈ 0.05 (near-zero exposure)
        #
        # Overnight budget: when temp > 1.0, cap position to limit gap risk
        # =====================================================================
        pos_strength_pre_risk_temp = pos_strength
        risk_temperature = 0.0
        risk_scale_factor = 1.0
        overnight_budget_applied = False
        overnight_max_position = None
        
        if RISK_TEMPERATURE_AVAILABLE:
            try:
                # Get cached risk temperature (avoids redundant API calls)
                risk_temp_result = get_cached_risk_temperature(
                    start_date="2020-01-01",
                    notional=NOTIONAL_PLN,
                    estimated_gap_risk=0.03,  # 3% default gap risk
                )
                
                # Apply scaling
                scaled_pos_strength, risk_meta = apply_risk_temperature_scaling(
                    pos_strength,
                    risk_temp_result,
                )
                
                # Extract values for Signal dataclass
                risk_temperature = risk_meta.get("risk_temperature", 0.0)
                risk_scale_factor = risk_meta.get("scale_factor", 1.0)
                overnight_budget_applied = risk_meta.get("overnight_budget_applied", False)
                overnight_max_position = risk_meta.get("overnight_max_position")
                
                # Update position strength
                pos_strength = scaled_pos_strength
                
            except Exception as e:
                # If risk temperature fails, continue with unscaled position
                if os.getenv("DEBUG"):
                    print(f"Risk temperature computation failed: {e}")

        # ================================================================
        # EXTRACT AUGMENTATION LAYER DATA FROM BMA METADATA
        # ================================================================
        # Hansen Skew-t (asymmetric return distribution)
        hansen_enabled = bma_meta.get("hansen_skew_t_enabled", False)
        hansen_lambda = bma_meta.get("hansen_lambda")
        hansen_nu = bma_meta.get("hansen_nu")
        hansen_skew_direction = bma_meta.get("hansen_skew_direction")
        
        # Contaminated Student-t (regime-dependent tails)
        cst_enabled = bma_meta.get("contaminated_student_t_enabled", False)
        cst_nu_normal = bma_meta.get("cst_nu_normal")
        cst_nu_crisis = bma_meta.get("cst_nu_crisis")
        cst_epsilon = bma_meta.get("cst_epsilon")

        # ================================================================
        # PIT VIOLATION EXIT SIGNAL (February 2026)
        # ================================================================
        # CORE DESIGN CONSTRAINT: "If the selected belief cannot be trusted,
        # the only correct signal is EXIT."
        #
        # This is BELIEF GOVERNANCE, not forecasting. EXIT means:
        # - Close existing positions
        # - Do not open new positions
        # - Await recalibration or regime change
        #
        # EXIT does NOT mean the stock will fall or another model is correct.
        # It means: "I no longer trust my belief about this stock."
        # ================================================================
        pit_exit_triggered = False
        pit_exit_reason = None
        pit_violation_severity = 0.0
        pit_penalty_effective = 1.0
        pit_selected_model = None
        
        if PIT_PENALTY_AVAILABLE:
            # Get selected model from BMA metadata
            pit_selected_model = bma_meta.get("dominant_model")
            if pit_selected_model is None:
                # Fallback to noise model from kalman metadata
                pit_selected_model = kalman_metadata.get("kalman_noise_model", "unknown")
            
            # Get PIT p-value for selected model
            # PREFER calibrated PIT p-value if isotonic recalibration was applied
            # Otherwise fall back to raw PIT p-value
            pit_pvalue_calibrated = kalman_metadata.get("pit_ks_pvalue_calibrated")
            pit_pvalue_raw = kalman_metadata.get("pit_ks_pvalue")
            
            if pit_pvalue_calibrated is not None and pit_pvalue_calibrated > 0:
                # Use calibrated PIT - recalibration "fixed" the distribution
                pit_pvalue_for_exit = pit_pvalue_calibrated
            else:
                # No recalibration applied - use raw PIT
                pit_pvalue_for_exit = pit_pvalue_raw
            
            # Get regime for threshold selection
            pit_regime = regime_used if regime_used is not None else -1
            
            # Compute PIT violation penalty for selected model
            if pit_pvalue_for_exit is not None:
                n_samples_for_pit = len(feats.get("returns", feats.get("px", [])))
                pit_result = compute_model_pit_penalty(
                    model_name=pit_selected_model,
                    pit_pvalue=pit_pvalue_for_exit,
                    regime=pit_regime,
                    n_samples=n_samples_for_pit,
                )
                
                pit_violation_severity = pit_result.violation_severity
                pit_penalty_effective = pit_result.effective_penalty
                pit_exit_triggered = pit_result.triggers_exit
                
                if pit_exit_triggered:
                    pit_exit_reason = (
                        f"Critical PIT violation: p={pit_pvalue_for_exit:.4f} "
                        f"< p_crit={pit_result.p_critical:.4f}, "
                        f"penalty={pit_penalty_effective:.3f} < {PIT_EXIT_THRESHOLD:.2f}"
                    )
                    # Override label to EXIT
                    label = "EXIT"
                    # Set position strength to 0 - no trading allowed
                    pos_strength = 0.0

        sigs.append(Signal(
            horizon_days=int(H),
            score=float(edge_now),
            p_up=float(p_now),
            exp_ret=float(mu_H),
            ci_low=ci_low,
            ci_high=ci_high,
            ci_low_90=ci_low_90,
            ci_high_90=ci_high_90,
            profit_pln=float(profit_pln),
            profit_ci_low_pln=float(profit_ci_low_pln),
            profit_ci_high_pln=float(profit_ci_high_pln),
            # ================================================================
            # POSITION STRENGTH FROM EXPECTED UTILITY (NOT KELLY)
            # ================================================================
            # All sizing is derived from the full posterior predictive
            # distribution (r_samples), not from point estimates.
            #
            # pos_strength = drift_weight × eu_position_size × risk_scale_factor
            # where eu_position_size = EU / max(E[loss], ε)
            #
            # This ensures:
            # - Fat downside tails → smaller positions
            # - Strong upside asymmetry → larger positions
            # - EU ≤ 0 → HOLD (position_strength = 0)
            # - High risk temperature → reduced exposure
            # ================================================================
            position_strength=float(pos_strength),
            vol_mean=float(vol_mean),
            vol_ci_low=float(vol_ci_low),
            vol_ci_high=float(vol_ci_high),
            regime=reg,
            label=label,
            # Expected Utility fields (THE BASIS FOR POSITION SIZING):
            expected_utility=float(expected_utility),
            expected_gain=float(expected_gain),
            expected_loss=float(expected_loss),
            gain_loss_ratio=float(gain_loss_ratio),
            eu_position_size=float(eu_position_size),
            # Risk Temperature fields (Expert Panel Solution 1 + 4):
            risk_temperature=float(risk_temperature),
            risk_scale_factor=float(risk_scale_factor),
            overnight_budget_applied=bool(overnight_budget_applied),
            overnight_max_position=float(overnight_max_position) if overnight_max_position is not None else None,
            pos_strength_pre_risk_temp=float(pos_strength_pre_risk_temp),
            # EVT (Extreme Value Theory) tail risk fields:
            expected_loss_empirical=float(expected_loss_empirical),
            evt_enabled=bool(evt_enabled),
            evt_xi=float(evt_xi) if evt_xi is not None else None,
            evt_sigma=float(evt_sigma) if evt_sigma is not None else None,
            evt_threshold=float(evt_threshold) if evt_threshold is not None else None,
            evt_n_exceedances=int(evt_n_exceedances),
            evt_fit_method=str(evt_fit_method) if evt_fit_method is not None else None,
            # Contaminated Student-t Mixture (regime-dependent tails):
            cst_enabled=bool(cst_enabled),
            cst_nu_normal=float(cst_nu_normal) if cst_nu_normal is not None else None,
            cst_nu_crisis=float(cst_nu_crisis) if cst_nu_crisis is not None else None,
            cst_epsilon=float(cst_epsilon) if cst_epsilon is not None else None,
            # Hansen Skew-t (asymmetric return distribution):
            hansen_enabled=bool(hansen_enabled),
            hansen_lambda=float(hansen_lambda) if hansen_lambda is not None else None,
            hansen_nu=float(hansen_nu) if hansen_nu is not None else None,
            hansen_skew_direction=str(hansen_skew_direction) if hansen_skew_direction is not None else None,
            # Diagnostics ONLY (NOT used for trading decisions):
            drift_uncertainty=float(drift_uncertainty_H),
            p_analytical=float(p_analytical),  # DIAGNOSTIC: analytical posterior predictive
            p_empirical=float(p_empirical),    # DIAGNOSTIC: raw empirical MC probability
            # STEP 7: Regime audit trace - tracks which regime params were used
            regime_used=int(regime_used) if regime_used is not None else None,
            regime_source=str(regime_source),
            regime_collapse_warning=bool(collapse_warning),
            # STEP 8: BMA audit trace - tracks model averaging method
            bma_method=str(bma_meta.get("method", "legacy")),
            bma_has_model_posterior=bool(bma_meta.get("has_bma", False)),
            bma_borrowed_from_global=bool(bma_meta.get("regime_details", {}).get(regime_used, {}).get("borrowed_from_global", False)) if regime_used is not None else False,
            # DUAL-SIDED TREND EXHAUSTION (market-space fragility):
            ue_up=float(ue_up),
            ue_down=float(ue_down),
            # PIT Violation EXIT Signal (February 2026):
            pit_exit_triggered=bool(pit_exit_triggered),
            pit_exit_reason=pit_exit_reason,
            pit_violation_severity=float(pit_violation_severity),
            pit_penalty_effective=float(pit_penalty_effective),
            pit_selected_model=pit_selected_model,
            # Volatility Estimator (February 2026):
            volatility_estimator=tuned_params.get('volatility_estimator') if tuned_params else None,
            # Enhanced Mixture (February 2026):
            mixture_enhanced=bool(ENHANCED_MIXTURE_ENABLED and ENHANCED_MIXTURE_AVAILABLE),
            # VIX-based ν adjustment (February 2026):
            vix_nu_adjustment_applied=bool(kalman_metadata.get('vix_nu_adjustment_applied', False)),
            nu_original=float(kalman_metadata.get('nu_original')) if kalman_metadata.get('nu_original') is not None else None,
            nu_adjusted=float(kalman_metadata.get('nu_adjusted')) if kalman_metadata.get('nu_adjusted') is not None else None,
            # CRPS-based model selection (February 2026):
            crps_score=float(tuned_params.get('crps')) if tuned_params and tuned_params.get('crps') is not None else None,
            scoring_weights=tuned_params.get('scoring_weights_used') if tuned_params else None,
            scoring_method=tuned_params.get('scoring_method') if tuned_params else None,
            # Story 3.6: MC Path Diagnostics
            mc_diagnostics=_mc_diag.to_dict() if _mc_diag and _mc_diag.n_paths > 0 else None,
            # Story 5.2: Balanced EU
            eu_asymmetric=float(EU_asymmetric),
            eu_balanced=float(EU_balanced),
            # Story 5.6: Kelly sizing
            kelly_full=float(kelly_full),
            kelly_half=float(kelly_half),
        ))

        # Story 3.3: Accumulate state for persistence
        _prev_signal_state[str(H)] = {
            "p_up": float(p_now),
            "label": label,
        }

    # Story 3.3: Save signal state for next run's two-day confirmation
    # Story 4.5: Also save smoothed regime probs for EMA continuity
    if asset_key:
        try:
            if regime_probs and isinstance(regime_probs, dict):
                _prev_signal_state["regime_probs"] = [
                    float(regime_probs.get(i, 0.2)) for i in range(5)
                ]
            save_signal_state(asset_key, _prev_signal_state)
        except Exception:
            pass  # non-critical: state persistence is best-effort

    # ── Story 7.5: Signal Output Validation Invariants ───────────────────
    _n_violations = 0
    _debug_mode = os.environ.get("SIGNAL_DEBUG", "") == "1"
    _sig_map: Dict[int, Signal] = {s.horizon_days: s for s in sigs}
    _prev_vol: Optional[float] = None
    for _h_idx, _H in enumerate(sorted(_sig_map.keys())):
        _sig = _sig_map[_H]
        # 1. Finiteness
        if not np.isfinite(_sig.exp_ret):
            _n_violations += 1
            if _debug_mode:
                raise AssertionError(f"exp_ret is not finite for H={_H}")
        if not np.isfinite(_sig.p_up):
            _n_violations += 1
            if _debug_mode:
                raise AssertionError(f"p_up is not finite for H={_H}")
        # 2. p_up in [0, 1]
        if not (0.0 <= _sig.p_up <= 1.0):
            _n_violations += 1
            if _debug_mode:
                raise AssertionError(f"p_up={_sig.p_up} out of [0,1] for H={_H}")
        # 3. CI ordering (allow small tolerance for numerical noise)
        if _sig.ci_low > _sig.ci_high + 1e-10:
            _n_violations += 1
            if _debug_mode:
                raise AssertionError(f"CI inversion at H={_H}: [{_sig.ci_low}, {_sig.ci_high}]")
        # 4. Non-negative volatility
        if hasattr(_sig, 'vol_mean') and _sig.vol_mean < 0:
            _n_violations += 1
            if _debug_mode:
                raise AssertionError(f"Negative vol_mean={_sig.vol_mean} for H={_H}")
        # 5. Variance monotonicity (allow 50% decrease tolerance)
        _curr_vol = _sig.vol_mean if hasattr(_sig, 'vol_mean') else 0.0
        if _prev_vol is not None and _prev_vol > 0 and _curr_vol > 0:
            if _curr_vol < _prev_vol * 0.5:
                _n_violations += 1
                if _debug_mode:
                    raise AssertionError(
                        f"vol decreased dramatically from H_prev to H={_H}: "
                        f"{_curr_vol:.6f} < {_prev_vol * 0.5:.6f}"
                    )
        _prev_vol = _curr_vol

    return sigs, thresholds


