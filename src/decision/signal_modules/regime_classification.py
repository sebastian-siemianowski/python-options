from __future__ import annotations
"""
Regime classification: asset type classification, horizon caps,
display price smoothing, and regime inference.

Extracted from signals.py (Story 5.3). Contains pure functions for
classifying assets by type, computing horizon-dependent forecast caps,
smoothing display prices, and inferring the current market regime.
"""
from typing import Dict, List, Optional, Tuple

import math

import numpy as np
import pandas as pd

from ingestion.data_utils import safe_last


# NOTE: No __all__ here. Without __all__, `from ... import *` exports all
# public names, including re-exported models.regime names (e.g.,
# compute_regime_probabilities_v2). Private names (_SIG_H_ANNUAL_CAP, etc.)
# are explicitly imported in signals.py.


#   - sig_H must be capped consistently so CI = mu_H ± z*sig_H stays bounded
#   - Asset-type-specific caps mirror market_temperature.py hard caps
#   - All values in LOG-RETURN space (multiply by 100 for %)
# ============================================================================

# Maximum sig_H per horizon per asset type (log-return units)
# Derived from realistic annual volatility maxima:
#   equity: ~120% annual → σ_daily ≈ 7.5% → sig_H(252) ≈ 1.20
#   currency: ~30% annual → σ_daily ≈ 1.9% → sig_H(252) ≈ 0.30
#   metal: ~60% annual → σ_daily ≈ 3.8% → sig_H(252) ≈ 0.60
#   crypto: ~200% annual → σ_daily ≈ 12.6% → sig_H(252) ≈ 2.00
_SIG_H_ANNUAL_CAP = {
    "equity": 1.20,    # 120% annual σ — covers high-vol small caps
    "currency": 0.30,  # 30% annual σ — covers EM currencies
    "metal": 0.60,     # 60% annual σ — covers volatile precious metals miners
    "crypto": 2.00,    # 200% annual σ — covers crypto
}

# Absolute CI bounds in log-return space (same limits as mu_H clamp)
_CI_LOG_FLOOR = -4.6    # exp(-4.6) - 1 ≈ -99%  (near-total loss)
_CI_LOG_CAP = 1.61      # exp(1.61) - 1 ≈ 400%  (5× gain)


def classify_asset_type(symbol: str) -> str:
    """Classify a ticker symbol into asset type for forecast bounding.

    Returns one of: 'equity', 'currency', 'metal', 'crypto'.

    Used by the signal pipeline to apply asset-type-aware CI bounds.
    Mirrors the classification in verify_forecasts.py and
    market_temperature.py hard caps.
    """
    if not symbol:
        return "equity"
    s = symbol.upper().strip()
    # Crypto detection: must be crypto pair format (BTC-USD) or exact ticker
    _CRYPTO_BASES = {"BTC", "ETH", "DOGE", "SOL", "ADA", "XRP", "AVAX",
                     "MATIC", "DOT", "LINK", "UNI", "SHIB"}
    if "-" in s:
        base = s.split("-")[0]
        if base in _CRYPTO_BASES:
            return "crypto"
    # Currency detection (FX pairs end in =X, or contain JPY/USD patterns)
    if s.endswith("=X"):
        return "currency"
    if "JPY" in s or "EUR" in s or "GBP" in s or "CHF" in s or "AUD" in s:
        # But not equities that happen to contain these strings
        if not any(s.startswith(p) for p in ["BTC", "ETH", "JPM"]):
            parts = s.split("=")
            if len(parts) > 1 or len(s) == 6:  # USDJPY=X or USDJPY
                return "currency"
    # Futures / metals
    if s.endswith("=F"):
        return "metal"
    _METAL_TICKERS = {
        "GLD", "SLV", "GDX", "GDXJ", "SIL", "SLVR", "GOLD", "NEM",
        "AEM", "WPM", "PAAS", "AG", "KGC", "FNV", "RGLD", "MAG",
        "EXK", "CDE", "HL", "FSM", "SILV",
    }
    if s in _METAL_TICKERS:
        return "metal"
    return "equity"


def _compute_sig_h_cap(H: int, asset_type: str) -> float:
    """Compute the maximum allowed sig_H for a given horizon and asset type.

    Uses √H scaling from the annual cap:
        sig_H_max = annual_cap × √(H / 252)

    A minimum floor ensures very short horizons don't over-tighten.
    """
    annual_cap = _SIG_H_ANNUAL_CAP.get(asset_type, 1.20)
    sig_cap = annual_cap * math.sqrt(max(H, 1) / 252.0)
    # Floor: at least 0.005 (0.5%) to avoid over-tightening short-horizon low-vol assets
    return max(sig_cap, 0.005)

# ============================================================================
# UPGRADE #3: Display Price Inertia (Presentation-Only)
# ============================================================================
# This cache stores previous display prices for presentation smoothing.
# Formula: display_price = 0.7 * prev_display_price + 0.3 * new_predicted_price
#
# IMPORTANT:
# - This does NOT affect trading decisions
# - This does NOT affect Expected Utility calculations
# - This does NOT affect regime detection
# - It ONLY prevents day-to-day jitter in displayed prices
#
# Institutions do this quietly for all client-facing price estimates.
# ============================================================================
_DISPLAY_PRICE_CACHE: Dict[Tuple[str, int], float] = {}
DISPLAY_PRICE_INERTIA = 0.3  # weight on previous display price (March 2026: reduced from 0.7 for responsiveness)


def _smooth_display_price(asset_key: str, horizon: int, new_price: float) -> float:
    """Apply presentation-only smoothing to predicted prices.
    
    Args:
        asset_key: Unique identifier for the asset (e.g., ticker symbol)
        horizon: Forecast horizon in days
        new_price: Newly computed predicted price
        
    Returns:
        Smoothed display price that reduces day-to-day jitter
    """
    cache_key = (asset_key, horizon)
    if cache_key in _DISPLAY_PRICE_CACHE:
        prev_price = _DISPLAY_PRICE_CACHE[cache_key]
        if np.isfinite(prev_price) and np.isfinite(new_price):
            smoothed = DISPLAY_PRICE_INERTIA * prev_price + (1.0 - DISPLAY_PRICE_INERTIA) * new_price
        else:
            smoothed = new_price if np.isfinite(new_price) else prev_price
    else:
        smoothed = new_price
    
    if np.isfinite(smoothed):
        _DISPLAY_PRICE_CACHE[cache_key] = smoothed
    
    return smoothed


def clear_display_price_cache() -> None:
    """Clear the display price inertia cache. Useful for testing or resets."""
    global _DISPLAY_PRICE_CACHE
    _DISPLAY_PRICE_CACHE.clear()


def infer_current_regime(feats: Dict[str, pd.Series], hmm_result: Optional[Dict] = None) -> Tuple[str, Dict[str, float]]:
    """
    Infer the current market regime using posterior inference from HMM.

    Args:
        feats: Feature dictionary
        hmm_result: Result from fit_hmm_regimes(), or None to use threshold fallback

    Returns:
        Tuple of (regime_label, regime_metadata_dict)
        regime_label: "calm", "trending", "crisis", or threshold-based fallback
        regime_metadata: probabilities and diagnostics
    """
    # If HMM available and fitted, use posterior inference
    if hmm_result is not None and "regime_series" in hmm_result:
        try:
            regime_series = hmm_result["regime_series"]
            posterior_probs = hmm_result["posterior_probs"]

            if not regime_series.empty:
                current_regime = regime_series.iloc[-1]
                current_probs = posterior_probs.iloc[-1].to_dict()

                return str(current_regime), {
                    "method": "hmm_posterior",
                    "probabilities": current_probs,
                    "persistence": float(hmm_result["transmat"][hmm_result["states"][-1], hmm_result["states"][-1]]) if len(hmm_result["states"]) > 0 else 0.5,
                }
        except Exception:
            pass

    # Fallback to threshold-based regime detection (original logic)
    vol_regime = feats.get("vol_regime", pd.Series(dtype=float))
    trend_z = feats.get("trend_z", pd.Series(dtype=float))

    vr = safe_last(vol_regime) if not vol_regime.empty else float("nan")
    tz = safe_last(trend_z) if not trend_z.empty else float("nan")

    # Threshold-based classification
    if np.isfinite(vr) and vr > 1.8:
        if np.isfinite(tz) and tz > 0:
            label = "High-vol uptrend"
        elif np.isfinite(tz) and tz < 0:
            label = "High-vol downtrend"
        else:
            label = "crisis"  # Map to HMM-style label
    elif np.isfinite(vr) and vr < 0.85:
        if np.isfinite(tz) and tz > 0:
            label = "Calm uptrend"
        elif np.isfinite(tz) and tz < 0:
            label = "Calm downtrend"
        else:
            label = "calm"  # Map to HMM-style label
    elif np.isfinite(tz) and abs(tz) > 0.5:
        label = "trending"
    else:
        label = "Normal"

    return label, {
        "method": "threshold_fallback",
        "vol_regime": float(vr) if np.isfinite(vr) else None,
        "trend_z": float(tz) if np.isfinite(tz) else None,
    }


# =============================================================================
# REGIME-CONDITIONAL BAYESIAN MODEL AVERAGING (RC-BMA)
# =============================================================================
# Implements: p(r_H | D) = Σ_r P(regime_r | D) · p(r_H | regime_r, D)
# Story 4.2: Regime functions imported from shared module models.regime
# =============================================================================
from models.regime import (
    REGIME_LOW_VOL_TREND, REGIME_HIGH_VOL_TREND, REGIME_LOW_VOL_RANGE,
    REGIME_HIGH_VOL_RANGE, REGIME_CRISIS_JUMP, REGIME_NAMES,
    CUSUM_THRESHOLD, CUSUM_COOLDOWN, CUSUM_ALPHA_ACCEL, CUSUM_ALPHA_NORMAL,
    _CUSUM_STATE, _get_cusum_state,
    assign_current_regime, map_regime_label_to_index,
    extract_regime_features, compute_regime_log_likelihoods,
    compute_regime_probabilities, compute_regime_probabilities_v2,
    compute_soft_bma_weights, _logistic,
    REGIME_TRANSITION_WIDTH_VOL, REGIME_TRANSITION_WIDTH_DRIFT,
    REGIME_VOL_HIGH_BOUNDARY, REGIME_VOL_LOW_BOUNDARY,
    REGIME_CRISIS_VOL_THRESHOLD, REGIME_CRISIS_TAIL_THRESHOLD,
    REGIME_CRISIS_TRANSITION_WIDTH,
    AdaptiveThresholds, compute_adaptive_thresholds,
    DRIFT_THRESHOLD_SIGMA, DEFAULT_DRIFT_THRESHOLD,
    DEFAULT_VOL_HIGH_BOUNDARY, DEFAULT_VOL_LOW_BOUNDARY,
    CUSUMParams, compute_cusum_params, compute_arl_threshold, decorrelation_time,
    REGIME_EMA_ALPHA, smooth_regime_probabilities,
)

# (Regime functions imported from models.regime via Story 4.2 above)

