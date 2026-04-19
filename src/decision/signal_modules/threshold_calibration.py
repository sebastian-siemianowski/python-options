"""Threshold calibration: edge composition, dynamic thresholds, confirmation logic.

Extracted from signals.py - Story 8.2.
Contains: composite_edge, optimal_threshold, compute_calibrated_thresholds,
          compute_dynamic_thresholds, compute_adaptive_edge_floor,
          apply_confirmation_logic.
"""

import math
import os
from typing import Dict, List, Optional

import numpy as np

# EDGE_FLOOR: transaction-cost/slippage hurdle (minimum absolute edge to act)
try:
    _edge_env = os.getenv("EDGE_FLOOR", "0.10")
    EDGE_FLOOR = float(_edge_env)
except Exception:
    EDGE_FLOOR = 0.10
EDGE_FLOOR = float(np.clip(EDGE_FLOOR, 0.0, 1.5))

# Story 5.1: Adaptive edge floor scaling factor
EDGE_FLOOR_Z = 0.65

def composite_edge(
    base_edge: float,
    trend_z: float,
    moms: List[float],
    vol_regime: float,
    z5: float,
) -> float:
    """Ensemble edge: blend trend-following and mean-reversion components.
    GARCH handles volatility dynamics; avoid extra regime dampening to prevent double-counting.
    """
    # Momentum confirmation: average tanh of t-momentum
    mom_terms = [np.tanh(m / 2.0) for m in moms if np.isfinite(m)]
    mom_align = float(np.mean(mom_terms)) if mom_terms else 0.0

    # Trend tilt (gentle)
    trend_tilt = float(np.tanh(trend_z / 2.0)) if np.isfinite(trend_z) else 0.0

    # TF component
    tf = base_edge + 0.30 * mom_align + 0.20 * trend_tilt

    # MR component: if z5 is very positive, expect mean-revert small negative edge; if very negative, mean-revert positive edge
    mr = float(-np.tanh(z5)) if np.isfinite(z5) else 0.0

    # Fixed blend (avoid vol_regime-driven dampening)
    w_tf, w_mr = 0.75, 0.25
    edge = w_tf * tf + w_mr * mr

    return float(edge)


# Story 5.4 ------------------------------------------------------------------
# Dynamic labeling thresholds from walk-forward hit rates
# -------------------------------------------------------------------------

def optimal_threshold(
    wf_records: list,
    horizon: int,
    target_hit_rate: float = 0.55,
    min_samples: int = 20,
) -> float:
    """Find the lowest p_up threshold achieving *target_hit_rate* on walk-forward data.

    Scans p_up in [0.50, 0.75) with 0.01 step.  Returns the first threshold where the
    subset's hit rate >= target.  Falls back to 0.55 if no threshold qualifies.
    """
    import pandas as pd

    if not wf_records:
        return 0.55

    df = pd.DataFrame([
        {"forecast_p_up": getattr(r, "forecast_p_up", r.get("forecast_p_up") if isinstance(r, dict) else None),
         "hit": getattr(r, "hit", r.get("hit") if isinstance(r, dict) else None),
         "horizon": getattr(r, "horizon", r.get("horizon") if isinstance(r, dict) else None)}
        for r in wf_records
    ])
    df = df[df["horizon"] == horizon].dropna(subset=["forecast_p_up", "hit"])

    if len(df) < min_samples:
        return 0.55

    for p_thresh in np.arange(0.50, 0.75, 0.01):
        mask = df["forecast_p_up"] > p_thresh
        if mask.sum() < min_samples:
            continue
        hit_rate = df.loc[mask, "hit"].mean()
        if hit_rate >= target_hit_rate:
            return float(round(p_thresh, 2))
    return 0.55


def compute_calibrated_thresholds(
    wf_records: list,
    horizons: Optional[List[int]] = None,
    target_hit_rate: float = 0.55,
) -> Dict[int, Dict[str, float]]:
    """Compute per-horizon buy/sell thresholds from walk-forward results.

    Returns ``{horizon: {"buy_thr": float, "sell_thr": float}}``
    where ``sell_thr = 1 - buy_thr`` (symmetric).
    """
    if horizons is None:
        horizons = [1, 3, 7, 21, 63]
    out: Dict[int, Dict[str, float]] = {}
    for h in horizons:
        buy = optimal_threshold(wf_records, h, target_hit_rate=target_hit_rate)
        sell = round(1.0 - buy, 2)
        out[h] = {"buy_thr": buy, "sell_thr": sell}
    return out


def compute_dynamic_thresholds(
    skew: float,
    regime_meta: Dict[str, float],
    sig_H: float,
    med_vol_last: float,
    H: int,
    calibrated_thresholds: Optional[Dict[int, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """
    Compute dynamic buy/sell thresholds with asymmetry and uncertainty adjustments.

    Story 5.4: If *calibrated_thresholds* is provided (from walk-forward hit rates),
    use those as starting points instead of the static base of 0.58/0.42.

    Args:
        skew: Return skewness (asymmetry measure)
        regime_meta: Regime detection metadata with method and probabilities
        sig_H: Forecast volatility at horizon H
        med_vol_last: Long-run median volatility
        H: Forecast horizon in days
        calibrated_thresholds: Optional per-horizon thresholds from walk-forward

    Returns:
        Dictionary with buy_thr, sell_thr, and uncertainty metrics
    """
    # Story 5.4: Use calibrated thresholds as base if available
    if calibrated_thresholds and H in calibrated_thresholds:
        _cal = calibrated_thresholds[H]
        base_buy = float(_cal.get("buy_thr", 0.58))
        base_sell = float(_cal.get("sell_thr", 0.42))
    else:
        base_buy, base_sell = 0.58, 0.42

    # Skew adjustment: shift thresholds based on return asymmetry
    g1 = float(np.clip(skew if np.isfinite(skew) else 0.0, -1.5, 1.5))
    skew_delta = 0.02 * float(np.tanh(abs(g1) / 0.75))

    if g1 < 0:  # Negative skew (crash risk)
        buy_thr = base_buy + skew_delta
        sell_thr = base_sell + skew_delta
    elif g1 > 0:  # Positive skew (rally potential)
        buy_thr = base_buy - skew_delta
        sell_thr = base_sell - skew_delta
    else:
        buy_thr, sell_thr = base_buy, base_sell

    # Regime-based uncertainty (HMM posterior entropy or vol regime deviation)
    if regime_meta.get("method") == "hmm_posterior":
        # Use Shannon entropy of regime posteriors as uncertainty measure
        probs = regime_meta.get("probabilities", {})
        entropy = 0.0
        for p in probs.values():
            if p > 1e-12:
                entropy -= p * np.log(p)
        # Normalize by max entropy (log(3) for 3 states)
        u_regime = float(np.clip(entropy / np.log(3.0), 0.0, 1.0))
    else:
        # Fallback: use vol_regime deviation if available
        vol_regime = regime_meta.get("vol_regime", 1.0)
        u_regime = float(np.clip(abs(vol_regime - 1.0) / 1.5, 0.0, 1.0)) if np.isfinite(vol_regime) else 0.5

    # Forecast uncertainty from realized vol vs historical
    med_sig_H = (med_vol_last * math.sqrt(H)) if (np.isfinite(med_vol_last) and med_vol_last > 0) else sig_H
    ratio = float(sig_H / med_sig_H) if med_sig_H > 0 else 1.0
    u_sig = float(np.clip(ratio - 1.0, 0.0, 1.0))

    # Combined uncertainty: regime entropy dominates, forecast uncertainty refines
    U = float(np.clip(0.5 * u_regime + 0.5 * u_sig, 0.0, 1.0))

    # Widen thresholds based on uncertainty
    widen_delta = 0.04 * U
    buy_thr += widen_delta
    sell_thr -= widen_delta

    # Clamp to reasonable ranges
    buy_thr = float(np.clip(buy_thr, 0.55, 0.70))
    sell_thr = float(np.clip(sell_thr, 0.30, 0.45))

    # Ensure minimum separation
    if buy_thr - sell_thr < 0.12:
        mid = 0.5
        sell_thr = min(sell_thr, mid - 0.06)
        buy_thr = max(buy_thr, mid + 0.06)

    return {
        "buy_thr": float(buy_thr),
        "sell_thr": float(sell_thr),
        "uncertainty": float(U),
        "u_regime": float(u_regime),
        "u_forecast": float(u_sig),
        "skew_adjustment": float(skew_delta),
    }


# Story 5.1 ------------------------------------------------------------------
def compute_adaptive_edge_floor(
    vol_daily: float,
    horizon: int,
    z: float = EDGE_FLOOR_Z,
    floor_min: float = 0.005,
    floor_max: float = 0.50,
) -> float:
    """Compute volatility-scaled edge floor.

    edge_floor = z * vol_annual / sqrt(H)

    Low-vol assets (currencies) get a smaller floor -> more signals.
    High-vol assets (crypto) get a larger floor -> higher bar.
    """
    if not np.isfinite(vol_daily) or vol_daily <= 0:
        return float(EDGE_FLOOR)
    vol_annual = vol_daily * math.sqrt(252)
    h_safe = max(horizon, 1)
    raw = z * vol_annual / math.sqrt(h_safe)
    return float(np.clip(raw, floor_min, floor_max))
# -----------------------------------------------------------------------------


def apply_confirmation_logic(
    p_smoothed_now: float,
    p_smoothed_prev: float,
    p_raw: float,
    pos_strength: float,
    buy_thr: float,
    sell_thr: float,
    edge: float,
    edge_floor: float
) -> str:
    """
    Apply 2-day confirmation with hysteresis to reduce signal churn.

    Level-7 modularization: Separates confirmation logic from main signal flow.

    Args:
        p_smoothed_now: Smoothed probability (current)
        p_smoothed_prev: Smoothed probability (previous)
        p_raw: Raw probability without smoothing
        pos_strength: Position strength (Expected Utility based, 0..1)
        buy_thr: Buy threshold
        sell_thr: Sell threshold
        edge: Composite edge score
        edge_floor: Minimum edge required to act

    Returns:
        Signal label: "STRONG BUY", "BUY", "HOLD", "SELL", or "STRONG SELL"
    """
    # Hysteresis bands (slightly wider than base thresholds)
    buy_enter = buy_thr + 0.01
    sell_enter = sell_thr - 0.01

    # Base label from 2-day confirmation (smoothed probabilities)
    label = "HOLD"
    if (p_smoothed_prev >= buy_enter) and (p_smoothed_now >= buy_enter):
        label = "BUY"
    elif (p_smoothed_prev <= sell_enter) and (p_smoothed_now <= sell_enter):
        label = "SELL"

    # Strong tiers based on raw conviction and EU-based position strength
    if p_raw >= max(0.66, buy_thr + 0.06) and pos_strength >= 0.30:
        label = "STRONG BUY"
    if p_raw <= min(0.34, sell_thr - 0.06) and pos_strength >= 0.30:
        label = "STRONG SELL"

    # Transaction-cost hurdle: force HOLD if absolute edge below floor
    if np.isfinite(edge) and abs(edge) < float(edge_floor):
        label = "HOLD"

    return label

