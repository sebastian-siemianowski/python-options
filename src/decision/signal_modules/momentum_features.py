from __future__ import annotations
"""
Momentum features: momentum scoring and dual-sided trend exhaustion.

Extracted from signals.py (Story 6.1). Contains compute_momentum_score
for multi-timeframe momentum scoring, compute_directional_exhaustion_from_features
for dual-sided trend exhaustion detection, and _compute_simple_exhaustion
for single-horizon exhaustion.
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -- path setup so "from ingestion..." works when run standalone ----------
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)



# ============================================================================
# DUAL-SIDED TREND EXHAUSTION (UE↑ / UE↓)
# ============================================================================
# Measures market-space trend fragility in two independent directions:
#   - Upside Exhaustion (UE↑): late-stage rally fragility, blow-off risk
#   - Downside Exhaustion (UE↓): late-stage sell-off fragility, rebound risk
#
# KEY DESIGN PRINCIPLES:
#   1. Exhaustion is directional in MARKET space, not position space
#   2. UE↑ and UE↓ are mutually exclusive (only one can be active)
#   3. Both bounded in (0, 1)
#   4. Neither decides direction - only modulates risk/confidence
#   5. No tuning feedback, no signal flips
#
# This removes the confusion of a single ambiguous "exhaustion" metric.
# ============================================================================




def compute_momentum_score(
    px_series: pd.Series,
    feats: Optional[Dict[str, pd.Series]] = None,
) -> int:
    """
    Compute momentum score from -100 (strong negative) to +100 (strong positive).
    
    Uses multi-timeframe momentum with volatility normalization:
    - Short-term (21d): Weight 0.4
    - Medium-term (63d): Weight 0.35
    - Long-term (126d): Weight 0.25
    
    Args:
        px_series: Price series
        feats: Optional features dict containing pre-computed momentum
        
    Returns:
        Integer momentum score from -100 to +100
    """
    if px_series is None or len(px_series) < 30:
        return 0
    
    try:
        # Get momentum from features if available
        if feats is not None:
            mom21 = feats.get("mom21")
            mom63 = feats.get("mom63")
            mom126 = feats.get("mom126")
            
            def get_last(s):
                if s is None:
                    return None
                if isinstance(s, pd.Series) and len(s) > 0:
                    return float(s.iloc[-1])
                return None
            
            m21 = get_last(mom21)
            m63 = get_last(mom63)
            m126 = get_last(mom126)
        else:
            m21 = m63 = m126 = None
        
        # Compute from price series if not in features
        if m21 is None and len(px_series) >= 21:
            ret21 = (px_series.iloc[-1] / px_series.iloc[-21]) - 1
            vol21 = px_series.pct_change().iloc[-21:].std() * np.sqrt(252)
            m21 = ret21 / max(vol21, 0.01) if vol21 > 0 else ret21 * 10
        
        if m63 is None and len(px_series) >= 63:
            ret63 = (px_series.iloc[-1] / px_series.iloc[-63]) - 1
            vol63 = px_series.pct_change().iloc[-63:].std() * np.sqrt(252)
            m63 = ret63 / max(vol63, 0.01) if vol63 > 0 else ret63 * 10
            
        if m126 is None and len(px_series) >= 126:
            ret126 = (px_series.iloc[-1] / px_series.iloc[-126]) - 1
            vol126 = px_series.pct_change().iloc[-126:].std() * np.sqrt(252)
            m126 = ret126 / max(vol126, 0.01) if vol126 > 0 else ret126 * 10
        
        # Compute weighted momentum score
        weights = []
        values = []
        
        if m21 is not None and np.isfinite(m21):
            weights.append(0.40)
            values.append(m21)
        if m63 is not None and np.isfinite(m63):
            weights.append(0.35)
            values.append(m63)
        if m126 is not None and np.isfinite(m126):
            weights.append(0.25)
            values.append(m126)
        
        if not values:
            return 0
        
        # Normalize weights
        total_weight = sum(weights)
        weighted_mom = sum(w * v for w, v in zip(weights, values)) / total_weight
        
        # Scale to -100 to +100
        # Typical vol-normalized momentum ranges from -3 to +3
        # Scale factor: multiply by 33 to get approximately -100 to +100
        scaled = weighted_mom * 33
        
        return int(np.clip(scaled, -100, 100))
        
    except Exception:
        return 0


def compute_directional_exhaustion_from_features(
    feats: Dict[str, pd.Series],
    lookback_short: int = 9,
    lookback_long: int = 21,
    vol_lookback: int = 21,
) -> Dict[str, float]:
    """
    Compute directional exhaustion as a 0-100% metric using multi-timeframe
    EMA analysis with Student-t fat-tail corrections.

    SENIOR QUANT PANEL METHODOLOGY:
    ===============================
    
    1. MULTI-TIMEFRAME DEVIATION ANALYSIS
       - Compute price deviation from 5 EMAs (9, 21, 50, 100, 200 days)
       - Separate into short-term (9, 21) and long-term (50, 100, 200) groups
       - Long-term deviation determines structural position
       - Short-term deviation determines recent move direction
    
    2. MOMENTUM ALIGNMENT DETECTION
       - Compare short-term vs long-term momentum (mom63 vs mom252)
       - Divergence indicates regime transition
       - Convergence indicates trend confirmation
    
    3. RECENT PEAK/TROUGH DETECTION
       - Find rolling 63-day high and low
       - Measure distance from recent extremes
       - Rally-then-breakdown: near recent high but falling
       - Capitulation-then-recovery: near recent low but rising
    
    4. FAT-TAIL PROBABILITY ADJUSTMENT
       - Use Student-t CDF instead of Gaussian
       - Heavy tails (low ν) → extreme moves more expected → lower exhaustion
    
    5. OUTPUT: 0-100% scale
       - ue_up > 0: Price above long-term equilibrium
       - ue_down > 0: Price below long-term equilibrium
       - Mutual exclusivity enforced

    Args:
        feats: Feature dictionary
        lookback_short, lookback_long, vol_lookback: Configuration params

    Returns:
        Dict with "ue_up" (0-1), "ue_down" (0-1), and diagnostics
    """
    from scipy.stats import norm as scipy_norm, t as scipy_t
    
    # Extract price series
    px_series = feats.get("px", pd.Series(dtype=float))
    if px_series is None or len(px_series) < 200:
        return _compute_simple_exhaustion(feats, lookback_short, lookback_long, vol_lookback)

    price = float(px_series.iloc[-1])
    if not np.isfinite(price) or price <= 0:
        return {"ue_up": 0.0, "ue_down": 0.0, "z_score": 0.0, "deviation_pct": 0.0}

    # =========================================================================
    # STEP 1: MULTI-TIMEFRAME EMA DEVIATIONS (SHORT vs LONG)
    # =========================================================================
    ema_periods_short = [9, 21]
    ema_periods_long = [50, 100, 200]
    
    ema_values = {}
    short_deviations = []
    long_deviations = []
    
    for period in ema_periods_short + ema_periods_long:
        if len(px_series) >= period:
            ema = px_series.ewm(span=period, adjust=False).mean()
            ema_now = float(ema.iloc[-1])
            if np.isfinite(ema_now) and ema_now > 0:
                deviation_pct = (price - ema_now) / ema_now
                ema_values[f"ema_{period}"] = ema_now
                if period in ema_periods_short:
                    short_deviations.append(deviation_pct)
                else:
                    long_deviations.append(deviation_pct)
    
    if not long_deviations:
        return {"ue_up": 0.0, "ue_down": 0.0, "z_score": 0.0, "deviation_pct": 0.0}
    
    # Calculate group averages
    short_dev_avg = np.mean(short_deviations) if short_deviations else 0.0
    long_dev_avg = np.mean(long_deviations)
    
    # Structural deviation (long-term determines direction)
    structural_deviation = long_dev_avg
    
    # =========================================================================
    # STEP 2: MOMENTUM EXTRACTION AND ALIGNMENT
    # =========================================================================
    mom21 = feats.get("mom21", pd.Series(dtype=float))
    mom63 = feats.get("mom63", pd.Series(dtype=float))
    mom126 = feats.get("mom126", pd.Series(dtype=float))
    mom252 = feats.get("mom252", pd.Series(dtype=float))
    
    def get_mom(m):
        if m is not None and len(m) > 0:
            v = float(m.iloc[-1])
            return v if np.isfinite(v) else 0.0
        return 0.0
    
    mom21_now = get_mom(mom21)
    mom63_now = get_mom(mom63)
    mom126_now = get_mom(mom126)
    mom252_now = get_mom(mom252)
    
    # Short-term vs long-term momentum comparison
    short_term_mom = (mom21_now + mom63_now) / 2
    long_term_mom = (mom126_now + mom252_now) / 2
    
    # Momentum alignment: positive = aligned, negative = diverging
    mom_alignment = short_term_mom * long_term_mom  # Same sign → positive product
    
    # =========================================================================
    # STEP 3: RECENT PEAK/TROUGH DETECTION (63-day lookback)
    # =========================================================================
    lookback_extreme = 63
    if len(px_series) >= lookback_extreme:
        rolling_high = float(px_series.iloc[-lookback_extreme:].max())
        rolling_low = float(px_series.iloc[-lookback_extreme:].min())
        
        # Distance from recent high (0 = at high, 1 = at low)
        high_low_range = rolling_high - rolling_low
        if high_low_range > 0:
            position_in_range = (price - rolling_low) / high_low_range
        else:
            position_in_range = 0.5
        
        # Detect breakdown from recent high
        distance_from_high_pct = (rolling_high - price) / rolling_high if rolling_high > 0 else 0
        distance_from_low_pct = (price - rolling_low) / rolling_low if rolling_low > 0 else 0
    else:
        position_in_range = 0.5
        distance_from_high_pct = 0.0
        distance_from_low_pct = 0.0
    
    # =========================================================================
    # STEP 4: VOLATILITY AND FAT-TAIL ADJUSTMENT
    # =========================================================================
    ret_series = feats.get("ret", pd.Series(dtype=float))
    if ret_series is not None and len(ret_series) >= vol_lookback:
        recent_vol = float(ret_series.iloc[-vol_lookback:].std())
        if not np.isfinite(recent_vol) or recent_vol <= 0:
            recent_vol = 0.02
    else:
        recent_vol = 0.02
    
    # Z-score based on structural deviation
    z_score = structural_deviation / max(recent_vol, 1e-10)
    
    # Get tail parameter
    nu_hat_series = feats.get("nu_hat", None)
    if nu_hat_series is not None and len(nu_hat_series) > 0:
        nu = float(nu_hat_series.iloc[-1])
        if not np.isfinite(nu) or nu <= 2:
            nu = 30.0
    else:
        nu = 30.0
    nu = max(4.0, min(nu, 100.0))
    
    # CDF transformation
    if nu < 30:
        cdf_val = scipy_t.cdf(abs(z_score), df=nu)
    else:
        cdf_val = scipy_norm.cdf(abs(z_score))
    
    exhaustion_base = 2.0 * (cdf_val - 0.5)  # Maps to (0, 1)
    
    # =========================================================================
    # STEP 5: PATTERN DETECTION
    # =========================================================================
    
    # EMA slope for trend direction
    ema_9_series = px_series.ewm(span=9, adjust=False).mean()
    if len(ema_9_series) >= 5:
        ema_9_slope = (float(ema_9_series.iloc[-1]) - float(ema_9_series.iloc[-5])) / max(float(ema_9_series.iloc[-5]), 1e-10)
    else:
        ema_9_slope = 0.0
    
    # Pattern 1: RALLY THEN BREAKDOWN
    # Long-term momentum strong positive, but short-term breaking down
    is_rally_breakdown = (
        long_term_mom > 1.0 and           # Strong long-term momentum
        short_term_mom < long_term_mom and # Short-term weakening
        ema_9_slope < -0.005 and          # 9-EMA turning down
        distance_from_high_pct > 0.05      # At least 5% off recent high
    )
    
    # Pattern 2: PARABOLIC RALLY (extreme)
    is_parabolic = (
        mom126_now > 2.0 and
        structural_deviation > 0.10
    )
    
    # Pattern 3: CAPITULATION
    is_capitulation = (
        mom63_now < -2.0 and
        mom126_now < -1.5 and
        structural_deviation < -0.15
    )
    
    # Pattern 4: RECOVERY FROM CRASH
    # Price recovering but still below long-term equilibrium
    is_recovery = (
        structural_deviation < 0 and       # Below long-term EMAs
        short_dev_avg > long_dev_avg and  # Short-term above long-term (recovering)
        ema_9_slope > 0.005               # Short-term trend up
    )
    
    # Pattern 5: PULLBACK IN UPTREND
    # Long-term trend up, short-term pullback
    is_pullback_uptrend = (
        long_term_mom > 0.5 and            # Long-term trend up
        structural_deviation > -0.05 and   # Not too far below
        short_dev_avg < 0 and              # Short-term below EMAs
        ema_9_slope < 0                    # Pulling back
    )
    
    # =========================================================================
    # STEP 6: FINAL CALCULATION WITH CONTEXT
    # =========================================================================
    
    if structural_deviation > 0:
        # PRICE ABOVE LONG-TERM EQUILIBRIUM → ue_up
        ue_up_raw = exhaustion_base
        
        # Boost for parabolic moves
        if is_parabolic:
            ue_up_raw = min(ue_up_raw * 1.4 + 0.15, 0.99)
        
        # Momentum confirmation boost
        if long_term_mom > 1.0:
            ue_up_raw = min(ue_up_raw + 0.1, 0.99)
        if mom252_now > 1.5:
            ue_up_raw = min(ue_up_raw + 0.1, 0.99)
        
        ue_up = min(ue_up_raw, 0.99)
        ue_down = 0.0
        
    elif structural_deviation < 0:
        # PRICE BELOW LONG-TERM EQUILIBRIUM → consider ue_down
        ue_down_raw = exhaustion_base
        
        # Rally-then-breakdown: this is MEAN REVERSION, not oversold
        # Flip to showing ue_up based on long-term momentum strength
        if is_rally_breakdown:
            # Strong prior rally means this breakdown is healthy
            if long_term_mom > 1.5:
                # Still structurally extended - show ue_up
                ue_up = min(0.25 + long_term_mom * 0.15, 0.70)
                ue_down = 0.0
                return {
                    "ue_up": float(ue_up),
                    "ue_down": 0.0,
                    "z_score": float(z_score),
                    "deviation_pct": float(structural_deviation * 100),
                    **ema_values,
                }
            else:
                # Moderate prior rally - reduce ue_down significantly
                ue_down_raw *= 0.3
        
        # Recovery pattern: reduce ue_down (price improving)
        if is_recovery:
            recovery_factor = 1.0 - min(short_dev_avg - long_dev_avg, 0.1) * 5
            ue_down_raw *= max(recovery_factor, 0.3)
        
        # Pullback in uptrend: show low ue_down (buying opportunity)
        if is_pullback_uptrend:
            ue_down_raw *= 0.4
            
        # Capitulation: boost ue_down
        if is_capitulation:
            ue_down_raw = min(ue_down_raw * 1.3 + 0.1, 0.99)
        
        # Momentum context penalty (positive long-term = less oversold)
        if not is_capitulation:
            if long_term_mom > 0.5:
                ue_down_raw *= 0.7
            if mom252_now > 1.0:
                ue_down_raw *= 0.6
        
        ue_down = min(max(ue_down_raw, 0.0), 0.99)
        ue_up = 0.0
        
    else:
        ue_up = 0.0
        ue_down = 0.0
    
    return {
        "ue_up": float(ue_up),
        "ue_down": float(ue_down),
        "z_score": float(z_score),
        "deviation_pct": float(structural_deviation * 100),
        **ema_values,
    }


def _compute_simple_exhaustion(
    feats: Dict[str, pd.Series],
    lookback_short: int = 9,
    lookback_long: int = 21,
    vol_lookback: int = 21,
) -> Dict[str, float]:
    """
    Fallback simple exhaustion calculation when not enough data for multi-timeframe.
    """
    from scipy.stats import norm as scipy_norm
    
    px_series = feats.get("px", pd.Series(dtype=float))
    if px_series is None or len(px_series) < max(lookback_short, lookback_long, vol_lookback):
        return {"ue_up": 0.0, "ue_down": 0.0, "z_score": 0.0, "deviation_pct": 0.0}

    price = float(px_series.iloc[-1])
    if not np.isfinite(price) or price <= 0:
        return {"ue_up": 0.0, "ue_down": 0.0, "z_score": 0.0, "deviation_pct": 0.0}

    ema_short = px_series.ewm(span=lookback_short, adjust=False).mean()
    ema_short_now = float(ema_short.iloc[-1])

    if not np.isfinite(ema_short_now) or ema_short_now <= 0:
        return {"ue_up": 0.0, "ue_down": 0.0, "z_score": 0.0, "deviation_pct": 0.0}

    ret_series = feats.get("ret", pd.Series(dtype=float))
    if ret_series is not None and len(ret_series) >= vol_lookback:
        recent_vol = float(ret_series.iloc[-vol_lookback:].std())
        if not np.isfinite(recent_vol) or recent_vol <= 0:
            recent_vol = 0.02
    else:
        recent_vol = 0.02

    deviation = price - ema_short_now
    deviation_pct = deviation / ema_short_now
    price_vol = ema_short_now * recent_vol
    z_score = deviation / max(price_vol, 1e-10)

    cdf_val = scipy_norm.cdf(abs(z_score))
    exhaustion_magnitude = 2.0 * (cdf_val - 0.5)

    if deviation > 0:
        ue_up = min(exhaustion_magnitude, 0.99)
        ue_down = 0.0
    elif deviation < 0:
        ue_up = 0.0
        ue_down = min(exhaustion_magnitude, 0.99)
    else:
        ue_up = 0.0
        ue_down = 0.0

    return {
        "ue_up": float(ue_up),
        "ue_down": float(ue_down),
        "z_score": float(z_score),
        "deviation_pct": float(deviation_pct * 100),
    }


