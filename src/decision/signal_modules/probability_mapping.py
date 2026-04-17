"""Probability mapping: calibration transport, EMOS correction, label assignment.

Extracted from signals.py - Story 8.3.
Contains: _load_signals_calibration, _apply_single_p_map, _apply_p_up_calibration,
          _apply_emos_correction, _apply_magnitude_bias_correction,
          _get_calibrated_label_thresholds, label_from_probability.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np


# ===========================================================================
# SIGNAL CALIBRATION CORRECTION (Pass 2 of Two-Pass Tuning)
# ===========================================================================
# When signals_calibration data exists in tuned_params, corrections are
# applied inline to p_up, exp_ret, and label thresholds.
# Graceful fallback: if missing, raw values are used (identity).
# ===========================================================================

def _load_signals_calibration(tuned_params: Optional[Dict]) -> Optional[Dict]:
    """Extract signals_calibration section from tuned params if available."""
    if tuned_params is None:
        return None
    cal = tuned_params.get("signals_calibration")
    if cal is None:
        return None
    v = cal.get("version", "")
    if v not in ("1.0", "2.0", "3.0", "3.1", "4.0", "5.0", "6.0"):
        return None
    return cal


def _apply_single_p_map(p_raw: float, p_map: Dict) -> Optional[float]:
    """Apply a single p_up calibration map (Beta, Platt, or isotonic).

    Helper for soft regime blending — applies one regime's calibration
    map without regime lookup logic.

    Args:
        p_raw:  Raw probability ∈ [0, 1]
        p_map:  Calibration map dict with 'type' key

    Returns:
        Calibrated probability ∈ [0, 1], or None if map is invalid.
    """
    map_type = p_map.get("type", "isotonic")
    if map_type == "beta":
        a = p_map.get("a", 1.0)
        b = p_map.get("b", 1.0)
        c = p_map.get("c", 0.0)
        p_clipped = max(0.005, min(0.995, p_raw))
        z = a * math.log(p_clipped) - b * math.log(1.0 - p_clipped) + c
        if z >= 0:
            result = 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            result = ez / (1.0 + ez)
        return max(0.0, min(1.0, result))
    elif map_type == "platt":
        a = p_map.get("a", 1.0)
        b = p_map.get("b", 0.0)
        p_clipped = max(0.01, min(0.99, p_raw))
        logit = math.log(p_clipped / (1.0 - p_clipped))
        z = a * logit + b
        if z >= 0:
            result = 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            result = ez / (1.0 + ez)
        return max(0.0, min(1.0, result))
    elif map_type == "isotonic":
        x = p_map.get("x", [0.0, 1.0])
        y = p_map.get("y", [0.0, 1.0])
        if len(x) < 2:
            return None
        result = float(np.interp(p_raw, x, y))
        return max(0.0, min(1.0, result))
    return None


def _apply_p_up_calibration(
    p_raw: float,
    cal: Optional[Dict],
    H: int,
    vol_regime: float = 1.0,
) -> float:
    """Apply p_up recalibration from signals calibration data.

    v7.0: Single Beta calibration layer — replaces 4-layer cascade.

    The old 4-layer cascade (Beta → Isotonic → TempScale → RegimeBlend)
    each shrunk p_up toward 0.5, compounding to ~10% signal loss:
      Raw p=0.62 → 0.59 (Beta) → 0.57 (Isotonic) → 0.56 (Temp) → 0.55 (Blend)

    Now: Single Beta map selected by closest regime centroid.
    No isotonic refinement, no temperature scaling, no softmax blending.

    Args:
        p_raw:      Raw p_up from BMA MC
        cal:        signals_calibration dict from tuned params
        H:          Horizon in days
        vol_regime: Current vol_regime for regime-conditional lookup

    Returns:
        Calibrated p_up (or raw if no calibration available)
    """
    if cal is None:
        return p_raw
    horizons = cal.get("horizons", {})
    h_cal = horizons.get(str(H))
    if h_cal is None:
        return p_raw

    # v7.0: Single Beta map — hard regime selection (no blending)
    by_regime = h_cal.get("by_regime")
    if by_regime is not None:
        _REGIME_CENTROIDS = {"LOW": 0.425, "NORMAL": 1.075, "HIGH": 2.0}

        # Find closest regime by distance to centroid
        best_regime = None
        best_dist = float('inf')
        for rname in list(_REGIME_CENTROIDS.keys()):
            if rname not in by_regime:
                continue
            centroid = _REGIME_CENTROIDS[rname]
            dist = abs(vol_regime - centroid)
            if dist < best_dist:
                best_dist = dist
                best_regime = rname

        # Fall back to ALL if no specific regime matched
        if best_regime is None:
            best_regime = "ALL"

        regime_cal = by_regime.get(best_regime, by_regime.get("ALL"))
        if regime_cal is not None:
            p_map = regime_cal.get("p_up_map")
            if p_map is not None:
                p_cal = _apply_single_p_map(p_raw, p_map)
                if p_cal is not None:
                    return max(0.0, min(1.0, p_cal))

    # Legacy fallback: top-level p_up_map (v2.0/v1.0)
    p_map = h_cal.get("p_up_map")
    if p_map is None:
        return p_raw

    map_type = p_map.get("type", "isotonic")

    if map_type == "beta":
        a = p_map.get("a", 1.0)
        b = p_map.get("b", 1.0)
        c = p_map.get("c", 0.0)
        p_clipped = max(0.005, min(0.995, p_raw))
        z = a * math.log(p_clipped) - b * math.log(1.0 - p_clipped) + c
        if z >= 0:
            result = 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            result = ez / (1.0 + ez)
        return max(0.0, min(1.0, result))

    if map_type == "platt":
        a = p_map.get("a", 1.0)
        b = p_map.get("b", 0.0)
        p_clipped = max(0.01, min(0.99, p_raw))
        logit = math.log(p_clipped / (1.0 - p_clipped))
        z = a * logit + b
        if z >= 0:
            result = 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            result = ez / (1.0 + ez)
        return max(0.0, min(1.0, result))

    # Legacy v1.0 isotonic map
    x = p_map.get("x", [0.0, 1.0])
    y = p_map.get("y", [0.0, 1.0])
    if len(x) < 2:
        return p_raw
    result = float(np.interp(p_raw, x, y))
    return max(0.0, min(1.0, result))


def _apply_emos_correction(
    mu_H: float,
    sigma_H: float,
    cal: Optional[Dict],
    H: int,
    vol_regime: float = 1.0,
) -> Tuple[float, float]:
    """Apply EMOS distributional correction to expected return and sigma.

    v5.0 CRITICAL FIX: EMOS parameters are fitted against percent-space
    data (pred * 100, actual * 100, sigma * 100) in the calibration
    walk-forward.  At inference, mu_H and sigma_H are in log-return
    units (e.g., 0.001 for 0.1%).  We must convert to percent before
    applying EMOS, then convert back.

    Without this fix, EMOS 'a' term (e.g., 0.018) adds 1.8% (should
    be 0.018%), and sigma 'c' term (e.g., 0.672) makes sig_H = 67.2%
    (should be 0.672%).  This inflates displayed returns and CI bounds
    by ~100x.

    v3.0: EMOS (Gneiting 2005) with regime conditioning.
    v2.0 fallback: mag_scale + bias (backward compat).

    Args:
        mu_H:       Raw expected return (log-return units)
        sigma_H:    Raw predicted sigma (log-return units)
        cal:        signals_calibration dict
        H:          Horizon in days
        vol_regime: Current vol_regime for regime lookup

    Returns:
        (mu_corrected, sigma_corrected) in log-return units
    """
    if cal is None:
        return mu_H, sigma_H
    horizons = cal.get("horizons", {})
    h_cal = horizons.get(str(H))
    if h_cal is None:
        return mu_H, sigma_H

    # v5.0: Convert to percent-space (matching calibration training data)
    mu_pct = mu_H * 100.0
    sig_pct = sigma_H * 100.0

    # v3.1: Soft regime blending for EMOS — same approach as p_up:
    # compute correction from each available regime, blend with
    # softmax weights based on distance from regime centroid.
    by_regime = h_cal.get("by_regime")
    if by_regime is not None:
        _REGIME_CENTROIDS = {"LOW": 0.425, "NORMAL": 1.075, "HIGH": 2.0}
        _BLEND_TAU = 0.2

        # Collect EMOS corrections from each available regime
        regime_corrections = {}
        for rname in list(_REGIME_CENTROIDS.keys()) + ["ALL"]:
            if rname not in by_regime:
                continue
            emos = by_regime[rname].get("emos")
            if emos is None or emos.get("type") != "emos":
                continue
            a = emos.get("a", 0.0)
            b = emos.get("b", 1.0)
            c = emos.get("c", 0.0)
            d = emos.get("d", 1.0)
            # v6.0: ν parameter available for Student-t predictive distribution
            # (stored but not used in the affine correction itself — the
            # correction formula is the same for Gaussian and Student-t EMOS)
            # nu = emos.get("nu")  # available for downstream Monte Carlo
            # Apply in percent-space (where EMOS was fitted)
            mu_cor_pct = a + b * mu_pct
            sig_cor_pct = max(0.01, c + d * sig_pct)
            regime_corrections[rname] = (mu_cor_pct, sig_cor_pct)

        if regime_corrections:
            specific = {k: v for k, v in regime_corrections.items() if k != "ALL"}
            if len(specific) >= 2:
                weights = {}
                for rname in specific:
                    centroid = _REGIME_CENTROIDS.get(rname, 1.0)
                    dist = abs(vol_regime - centroid)
                    weights[rname] = math.exp(-dist / _BLEND_TAU)
                w_sum = sum(weights.values())
                if w_sum > 0:
                    mu_blend = sum(weights[r] / w_sum * specific[r][0] for r in specific)
                    sig_blend = sum(weights[r] / w_sum * specific[r][1] for r in specific)
                    # Convert back to log-return units
                    return mu_blend / 100.0, max(1e-6, sig_blend / 100.0)
            if specific:
                best_r = min(specific.keys(),
                             key=lambda r: abs(vol_regime - _REGIME_CENTROIDS.get(r, 1.0)))
                mu_c, sig_c = specific[best_r]
                return mu_c / 100.0, max(1e-6, sig_c / 100.0)
            if "ALL" in regime_corrections:
                mu_c, sig_c = regime_corrections["ALL"]
                return mu_c / 100.0, max(1e-6, sig_c / 100.0)

    # Legacy v2.0 fallback: mag_scale + bias
    mag_scale = h_cal.get("mag_scale", 1.0)
    bias = h_cal.get("bias", 0.0)
    corrected = mu_H * mag_scale + bias / 100.0
    return corrected, sigma_H


def _apply_magnitude_bias_correction(
    mu_H: float, cal: Optional[Dict], H: int
) -> float:
    """Legacy v2.0 magnitude+bias correction (kept for backward compat).

    v3.0 assets use _apply_emos_correction() instead.

    Args:
        mu_H: Raw expected return (log-return units)
        cal: signals_calibration dict from tuned params
        H: Horizon in days

    Returns:
        Corrected expected return
    """
    if cal is None:
        return mu_H
    horizons = cal.get("horizons", {})
    h_cal = horizons.get(str(H))
    if h_cal is None:
        return mu_H
    mag_scale = h_cal.get("mag_scale", 1.0)
    bias = h_cal.get("bias", 0.0)
    corrected = mu_H * mag_scale + bias / 100.0
    return corrected


def _get_calibrated_label_thresholds(
    cal: Optional[Dict],
    H: int = 7,
) -> Optional[Tuple[float, float]]:
    """Get per-asset label thresholds from calibration data.

    v2.0: per-horizon thresholds with fallback to global.

    Args:
        cal: signals_calibration dict
        H: Horizon in days (used for per-horizon lookup in v2.0)

    Returns:
        (buy_thr, sell_thr) or None if not available
    """
    if cal is None:
        return None

    # v2.0: try per-horizon thresholds first
    by_horizon = cal.get("label_thresholds_by_horizon", {})
    h_thr = by_horizon.get(str(H))
    if h_thr is not None:
        buy_thr = h_thr.get("buy_thr")
        sell_thr = h_thr.get("sell_thr")
        if buy_thr is not None and sell_thr is not None:
            return (float(buy_thr), float(sell_thr))

    # Fallback to global label_thresholds (v1.0 compat)
    label_thr = cal.get("label_thresholds")
    if label_thr is None:
        return None
    buy_thr = label_thr.get("buy_thr")
    sell_thr = label_thr.get("sell_thr")
    if buy_thr is None or sell_thr is None:
        return None
    return (float(buy_thr), float(sell_thr))


def label_from_probability(p_up: float, pos_strength: float, buy_thr: float = 0.58, sell_thr: float = 0.42) -> str:
    """Map probability and position strength to label with customizable thresholds.
    - STRONG tiers require both probability and position_strength to be high.
    - buy_thr and sell_thr must satisfy sell_thr < 0.5 < buy_thr.
    - pos_strength is derived from Expected Utility (EU / max(E[loss], ε))
    """
    buy_thr = float(buy_thr)
    sell_thr = float(sell_thr)
    # Strong tiers (EU-based sizing: 0.30 threshold for strong conviction)
    if p_up >= max(0.66, buy_thr + 0.06) and pos_strength >= 0.30:
        return "STRONG BUY"
    if p_up <= min(0.34, sell_thr - 0.06) and pos_strength >= 0.30:
        return "STRONG SELL"
    # Base labels
    if p_up >= buy_thr:
        return "BUY"
    if p_up <= sell_thr:
        return "SELL"
    return "HOLD"


