#!/usr/bin/env python3
"""
CSI BATCH 5-7: v53-v72 — Exploiting v49 + Deep Combinations
============================================================
NEW CHAMPION: v49 Consec Patterns (C=0.586, Sh=0.556, Spr=+5.5%, GS=55/98)

Strategy:
- Batch 5 (v53-v62): v49 variants + hybrids with MFI/Fib
- Batch 6 (v63-v72): Ensemble methods + parameter sweeps
"""

import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi_mega_harness import (
    UNIVERSE, preload_all, get_indicators, evaluate, summarize, run_strategy,
    v8_baseline, v17_mfi_divergence, LEADERBOARD_FILE,
)


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def _count_runs(series):
    """Count consecutive 1s, reset on 0. Returns numpy array."""
    vals = series.values.astype(float)
    out = np.zeros(len(vals))
    for i in range(1, len(vals)):
        if vals[i] > 0:
            out[i] = out[i-1] + 1
    return pd.Series(out, index=series.index)


def _mfi_divergence(ind):
    """Compute MFI divergence signals. Returns (bullish_div, bearish_div)."""
    close = ind["close"]; mfi = ind["mfi"]
    price_high = (close >= close.rolling(20).max() * 0.98).astype(float)
    mfi_weak = (mfi < mfi.rolling(20).max() * 0.90).astype(float)
    bearish_div = price_high * mfi_weak
    price_low = (close <= close.rolling(20).min() * 1.02).astype(float)
    mfi_strong = (mfi > mfi.rolling(20).min() * 1.10).astype(float)
    bullish_div = price_low * mfi_strong
    return bullish_div, bearish_div


# ═══════════════════════════════════════════════════════════
# BATCH 5: v53-v62 — v49 Variants + Hybrids
# ═══════════════════════════════════════════════════════════

def v49_base(ind):
    """v49 reimplemented for reference."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mr_sig = dn_exhaust * 0.5 - up_exhaust * 0.5
    mom_sig = up_mom * 0.3 - dn_mom * 0.3
    raw = (mr_sig + mom_sig + 0.30 * ind["vol_flow"] + 0.20 * ind["trend_score"]) * 80
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v53_consec_mfi_sell(ind):
    """v53: v49 base + MFI sell detection overlay."""
    close = ind["close"]; mfi = ind["mfi"]
    base = v49_base(ind)

    # MFI bearish divergence
    _, bearish_div = _mfi_divergence(ind)
    mfi_sell = pd.Series(np.where(mfi < 45, (45 - mfi) / 45, 0.0), index=close.index).clip(0, 1)

    # Overlay: reduce signal when MFI signals distribution
    sell_pressure = 0.6 * bearish_div + 0.4 * mfi_sell
    result = base - sell_pressure * 15
    return result.clip(-100, 100)


def v54_consec_mfi_full(ind):
    """v54: v49 + full MFI signal (both buy and sell sides)."""
    close = ind["close"]; mfi = ind["mfi"]
    base = v49_base(ind)
    bullish_div, bearish_div = _mfi_divergence(ind)
    mfi_sig = (mfi - 50) / 50

    # Add MFI as overlay
    mfi_overlay = mfi_sig * 15 + bullish_div * 10 - bearish_div * 12
    result = base + mfi_overlay
    return result.clip(-100, 100).ewm(span=2, adjust=False).mean()


def v55_consec_tuned(ind):
    """v55: v49 with tuned thresholds (3+ instead of 4+ for exhaustion)."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)

    # Lower exhaustion threshold: 3+ days
    up_exhaust = (up_runs >= 3).astype(float) * ((up_runs - 2) / 4).clip(0, 1)
    dn_exhaust = (dn_runs >= 3).astype(float) * ((dn_runs - 2) / 4).clip(0, 1)

    # Continuation: exactly 2 days
    up_mom = (up_runs == 2).astype(float)
    dn_mom = (dn_runs == 2).astype(float)

    mr_sig = dn_exhaust * 0.5 - up_exhaust * 0.5
    mom_sig = up_mom * 0.3 - dn_mom * 0.3
    raw = (mr_sig + mom_sig + 0.30 * ind["vol_flow"] + 0.20 * ind["trend_score"]) * 80
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v56_consec_vol_weight(ind):
    """v56: v49 with volume-weighted runs (high volume runs count more)."""
    close = ind["close"]; volume = ind["volume"]
    vol_rel = ind["vol_rel"]  # volume / 20d avg

    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)

    # Volume conviction: high volume runs are more meaningful
    vol_conviction = vol_rel.rolling(3).mean().clip(0.5, 2.0)

    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1) * vol_conviction
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1) * vol_conviction

    # Low volume exhaustion = weak signal (might continue)
    # High volume exhaustion = strong reversal signal
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float) * vol_conviction.clip(0, 1.5)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float) * vol_conviction.clip(0, 1.5)

    mr_sig = dn_exhaust * 0.5 - up_exhaust * 0.5
    mom_sig = up_mom * 0.3 - dn_mom * 0.3
    raw = (mr_sig + mom_sig + 0.25 * ind["vol_flow"] + 0.20 * ind["trend_score"]) * 80
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v57_consec_magnitude(ind):
    """v57: v49 using magnitude of consecutive moves, not just direction."""
    close = ind["close"]
    ret = ind["ret_1"]

    # Cumulative return during runs
    up = (ret > 0).astype(float)
    dn = (ret < 0).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)

    # Cumulative magnitude during current run
    def cum_run_return(returns, run_counts):
        """Accumulate returns during consecutive runs."""
        vals = returns.values.astype(float)
        runs = run_counts.values.astype(float)
        out = np.zeros(len(vals))
        for i in range(len(vals)):
            if runs[i] > 0:
                out[i] = out[i-1] + vals[i] if i > 0 else vals[i]
            else:
                out[i] = 0
        return pd.Series(out, index=returns.index)

    up_cum = cum_run_return(ret, up_runs)
    dn_cum = cum_run_return(ret, dn_runs)

    # Large cumulative moves = exhaustion
    vol_20 = ind["vol_20"].replace(0, float("nan"))
    up_z = (up_cum / (vol_20 * np.sqrt(up_runs.clip(1)))).clip(0, 5)
    dn_z = (dn_cum.abs() / (vol_20 * np.sqrt(dn_runs.clip(1)))).clip(0, 5)

    # Exhaustion: > 2 sigma cumulative move
    up_exhaust = (up_z > 1.5).astype(float) * ((up_z - 1.5) / 2).clip(0, 1)
    dn_exhaust = (dn_z > 1.5).astype(float) * ((dn_z - 1.5) / 2).clip(0, 1)

    # Momentum: 1-2 sigma in direction
    up_mom = ((up_z > 0.5) & (up_z <= 1.5) & (up_runs >= 2)).astype(float)
    dn_mom = ((dn_z > 0.5) & (dn_z <= 1.5) & (dn_runs >= 2)).astype(float)

    mr_sig = dn_exhaust * 0.5 - up_exhaust * 0.5
    mom_sig = up_mom * 0.3 - dn_mom * 0.3
    raw = (mr_sig + mom_sig + 0.30 * ind["vol_flow"] + 0.20 * ind["trend_score"]) * 80
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v58_consec_v8corr(ind):
    """v58: v49 + v8 correction suite."""
    close = ind["close"]; rsi = ind["rsi"]
    above_200 = ind["above_200"].fillna(0)
    base = v49_base(ind)

    # v8 corrections
    stoch_k = ind["stoch_k"]
    oversold_c = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    stoch_os = pd.Series(np.where(stoch_k < 20, (20 - stoch_k) / 20, 0.0), index=close.index).clip(0, 1)
    os_strength = np.maximum(oversold_c, stoch_os)
    base = base + os_strength * above_200 * 30 - os_strength * (1 - above_200) * 8

    rsi_ob = pd.Series(np.where(rsi > 68, (rsi - 68) / 32, 0.0), index=close.index).clip(0, 1)
    base = base - rsi_ob * (ind["ma50_slope"] < 0).astype(float) * 20

    # Volume divergence at highs
    price_near_high = ((close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min()).replace(0, float("nan"))).clip(0, 1)
    vol_10 = ind["volume"].rolling(10).mean()
    vol_30 = ind["volume"].rolling(30).mean()
    base = base - (price_near_high > 0.70).astype(float) * ((vol_30 - vol_10) / vol_30.replace(0, float("nan"))).clip(0, 1) * 8

    return base.clip(-100, 100)


def v59_consec_trend_filtered(ind):
    """v59: v49 with stronger trend filter — only mean-revert in uptrends."""
    close = ind["close"]
    above_200 = ind["above_200"].fillna(0)
    above_50 = ind["above_50"]

    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)

    # Exhaustion: only mean-revert oversold in uptrends
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)

    # In uptrend: dn_exhaust = buy signal, up_exhaust = weak sell
    # In downtrend: dn_exhaust = could continue, up_exhaust = sell
    up_trend = above_200 * 0.5 + above_50 * 0.5

    mr_buy = dn_exhaust * up_trend * 0.6  # Buy oversold dips in uptrend
    mr_sell = up_exhaust * 0.3 + dn_exhaust * (1 - up_trend) * (-0.2)  # Sell exhaustion + downtrend continuation

    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float) * 0.3
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float) * 0.3

    raw = (mr_buy - mr_sell + up_mom - dn_mom + 0.25 * ind["vol_flow"] + 0.20 * ind["trend_score"]) * 80
    raw = raw + above_200 * 5 + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v60_consec_fib(ind):
    """v60: v49 + Fibonacci position (combine two positive-spread strategies)."""
    close = ind["close"]; high = ind["high"]; low = ind["low"]

    # v49 component
    base = v49_base(ind)

    # Fib component (from v48)
    h40 = high.rolling(40).max()
    l40 = low.rolling(40).min()
    swing = (h40 - l40).replace(0, float("nan"))
    range_pos = ((close - l40) / swing).clip(0, 1)
    fib_sig = -(range_pos - 0.5) * 2  # Buy low, sell high

    # Blend: 70% v49 + 30% Fib
    raw = 0.70 * base + 0.30 * fib_sig * 60
    return raw.clip(-100, 100).ewm(span=2, adjust=False).mean()


def v61_consec_regime(ind):
    """v61: v49 with regime-adaptive weights (more aggressive in high ADX)."""
    close = ind["close"]
    adx_r = ind["adx_regime"].fillna(0.5)

    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)

    # In trending regime: continuation signals stronger
    # In ranging regime: exhaustion signals stronger
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)

    # Regime-adaptive: more mean-reversion when ranging, more momentum when trending
    mr_w = 1 - adx_r  # High when ranging
    mom_w = adx_r  # High when trending

    mr_sig = (dn_exhaust * 0.5 - up_exhaust * 0.5) * (0.5 + 0.5 * mr_w)
    mom_sig = (up_mom * 0.3 - dn_mom * 0.3) * (0.5 + 0.5 * mom_w)

    raw = (mr_sig + mom_sig + 0.25 * ind["vol_flow"] + 0.20 * ind["trend_score"]) * 85
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v62_consec_multi_scale(ind):
    """v62: Multi-scale consecutive patterns (1d, 2d, 3d bars)."""
    close = ind["close"]
    above_200 = ind["above_200"].fillna(0)

    def _pattern_signal(close_series, scale=1):
        """Compute pattern signal at given scale (1=daily, 2=2day, etc.)."""
        if scale > 1:
            c = close_series.iloc[::scale].reindex(close_series.index, method="ffill")
        else:
            c = close_series
        up = (c > c.shift(1)).astype(float)
        dn = (c < c.shift(1)).astype(float)
        up_r = _count_runs(up)
        dn_r = _count_runs(dn)
        up_ex = (up_r >= 4).astype(float) * ((up_r - 3) / 3).clip(0, 1)
        dn_ex = (dn_r >= 4).astype(float) * ((dn_r - 3) / 3).clip(0, 1)
        up_m = ((up_r >= 2) & (up_r <= 3)).astype(float)
        dn_m = ((dn_r >= 2) & (dn_r <= 3)).astype(float)
        return (dn_ex * 0.5 - up_ex * 0.5) + (up_m * 0.3 - dn_m * 0.3)

    s1 = _pattern_signal(close, 1)  # Daily
    s2 = _pattern_signal(close, 2)  # 2-day bars
    s3 = _pattern_signal(close, 3)  # 3-day bars

    raw = (0.50 * s1 + 0.30 * s2 + 0.20 * s3 + 0.20 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 80
    raw = raw + above_200 * 5 + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


# ═══════════════════════════════════════════════════════════
# BATCH 6: v63-v72 — Ensemble + Parameter Sweeps
# ═══════════════════════════════════════════════════════════

def v63_consec_ema_slow(ind):
    """v63: v49 with slower EMA smoothing (span=5 instead of 3)."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mr_sig = dn_exhaust * 0.5 - up_exhaust * 0.5
    mom_sig = up_mom * 0.3 - dn_mom * 0.3
    raw = (mr_sig + mom_sig + 0.30 * ind["vol_flow"] + 0.20 * ind["trend_score"]) * 80
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100).ewm(span=5, adjust=False).mean()


def v64_consec_no_smooth(ind):
    """v64: v49 with no EMA smoothing (raw signal)."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mr_sig = dn_exhaust * 0.5 - up_exhaust * 0.5
    mom_sig = up_mom * 0.3 - dn_mom * 0.3
    raw = (mr_sig + mom_sig + 0.30 * ind["vol_flow"] + 0.20 * ind["trend_score"]) * 80
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v65_consec_heavy_trend(ind):
    """v65: v49 with heavier trend weight (40% instead of 20%)."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mr_sig = dn_exhaust * 0.5 - up_exhaust * 0.5
    mom_sig = up_mom * 0.3 - dn_mom * 0.3
    raw = (mr_sig + mom_sig + 0.20 * ind["vol_flow"] + 0.40 * ind["trend_score"]) * 80
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v66_consec_light_flow(ind):
    """v66: v49 with less volume flow weight (15% instead of 30%)."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mr_sig = dn_exhaust * 0.6 - up_exhaust * 0.6
    mom_sig = up_mom * 0.35 - dn_mom * 0.35
    raw = (mr_sig + mom_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v67_consec_stronger_mr(ind):
    """v67: v49 with stronger mean-reversion weights (0.7 instead of 0.5)."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mr_sig = dn_exhaust * 0.7 - up_exhaust * 0.7
    mom_sig = up_mom * 0.2 - dn_mom * 0.2
    raw = (mr_sig + mom_sig + 0.30 * ind["vol_flow"] + 0.20 * ind["trend_score"]) * 80
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v68_consec_higher_bias(ind):
    """v68: v49 with higher bias (+3 instead of +1 from above_200*5)."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mr_sig = dn_exhaust * 0.5 - up_exhaust * 0.5
    mom_sig = up_mom * 0.3 - dn_mom * 0.3
    raw = (mr_sig + mom_sig + 0.30 * ind["vol_flow"] + 0.20 * ind["trend_score"]) * 80
    raw = raw + ind["above_200"].fillna(0) * 8 + 3
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v69_consec_lower_bias(ind):
    """v69: v49 with lower bias (above_200*3, no constant)."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mr_sig = dn_exhaust * 0.5 - up_exhaust * 0.5
    mom_sig = up_mom * 0.3 - dn_mom * 0.3
    raw = (mr_sig + mom_sig + 0.30 * ind["vol_flow"] + 0.20 * ind["trend_score"]) * 80
    raw = raw + ind["above_200"].fillna(0) * 3
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v70_ensemble_v49_v17(ind):
    """v70: 60/40 ensemble of v49 + v17 (top two strategies)."""
    s49 = v49_base(ind)
    s17 = v17_mfi_divergence(ind)
    return (0.60 * s49 + 0.40 * s17).clip(-100, 100)


def v71_ensemble_v49_v48(ind):
    """v71: 70/30 ensemble v49 + v48 Fib (best spread + best GS)."""
    close = ind["close"]; high = ind["high"]; low = ind["low"]
    s49 = v49_base(ind)

    # v48 fib signal
    h40 = high.rolling(40).max()
    l40 = low.rolling(40).min()
    swing = (h40 - l40).replace(0, float("nan"))
    range_pos = ((close - l40) / swing).clip(0, 1)
    fib_sig = -(range_pos - 0.5) * 2 * 40  # Scale to similar range

    return (0.70 * s49 + 0.30 * fib_sig).clip(-100, 100).ewm(span=2, adjust=False).mean()


def v72_mega_ensemble(ind):
    """v72: Weighted ensemble of top 5 all-time strategies."""
    s49 = v49_base(ind)                 # C=0.586
    s17 = v17_mfi_divergence(ind)       # C=0.526
    # v48 fib inline
    close = ind["close"]; high = ind["high"]; low = ind["low"]
    h40 = high.rolling(40).max(); l40 = low.rolling(40).min()
    swing = (h40 - l40).replace(0, float("nan"))
    range_pos = ((close - l40) / swing).clip(0, 1)
    fib_sig = -(range_pos - 0.5) * 2 * 40
    # v35 vol-adj mfi inline
    mfi = ind["mfi"]; vol_pct = ind["vol_pct"].fillna(0.5)
    mfi_trend = (mfi - 50) / 50
    vol_adj = mfi_trend * 50 * (1.1 - 0.2 * vol_pct)

    # Weights by Combined score
    total = 0.586 + 0.526 + 0.488 + 0.526
    result = (0.586/total * s49 + 0.526/total * s17 + 0.488/total * fib_sig + 0.526/total * vol_adj)
    return result.clip(-100, 100).ewm(span=2, adjust=False).mean()


def main():
    print("=" * 110)
    print("  CSI BATCH 5-7: v53-v72 — v49 Variants + Hybrids + Parameter Sweeps")
    print("=" * 110)

    t0 = time.time()
    n = preload_all()
    print(f"\n  Loaded {n} assets, computing indicators...")
    for sym in UNIVERSE:
        get_indicators(sym)
    print(f"  Ready in {time.time()-t0:.1f}s")

    print(f"\n  {'Strategy':<30} {'Sharpe':>6} {'Spread':>6} {'Comb':>6} "
          f"{'BuyP':>7} {'SellC':>7} {'SH%':>5} {'GoodSep':>10} {'Mkt':>4} {'DD':>6}")
    print(f"  {'='*30} {'='*6} {'='*6} {'='*6} {'='*7} {'='*7} {'='*5} {'='*10} {'='*4} {'='*6}")

    strategies = [
        ("v49 ConsecPat (champion)", v49_base),
        ("v17 MFI (prev champ)", v17_mfi_divergence),
        # Batch 5: v49 hybrids
        ("v53 Consec+MFI Sell", v53_consec_mfi_sell),
        ("v54 Consec+MFI Full", v54_consec_mfi_full),
        ("v55 Consec Tuned 3+", v55_consec_tuned),
        ("v56 Consec VolWeight", v56_consec_vol_weight),
        ("v57 Consec Magnitude", v57_consec_magnitude),
        ("v58 Consec+v8Corr", v58_consec_v8corr),
        ("v59 Consec TrendFilt", v59_consec_trend_filtered),
        ("v60 Consec+Fib", v60_consec_fib),
        ("v61 Consec Regime", v61_consec_regime),
        ("v62 Consec MultiScale", v62_consec_multi_scale),
        # Batch 6: Parameter sweeps + ensembles
        ("v63 Consec EMA5", v63_consec_ema_slow),
        ("v64 Consec NoSmooth", v64_consec_no_smooth),
        ("v65 Consec HeavyTrend", v65_consec_heavy_trend),
        ("v66 Consec LightFlow", v66_consec_light_flow),
        ("v67 Consec StrongMR", v67_consec_stronger_mr),
        ("v68 Consec HighBias", v68_consec_higher_bias),
        ("v69 Consec LowBias", v69_consec_lower_bias),
        ("v70 Ensemble v49+v17", v70_ensemble_v49_v17),
        ("v71 Ensemble v49+v48", v71_ensemble_v49_v48),
        ("v72 Mega Ensemble Top5", v72_mega_ensemble),
    ]

    all_results = []
    for name, fn in strategies:
        t = time.time()
        s = run_strategy(name, fn)
        s["time"] = round(time.time() - t, 1)
        all_results.append(s)

    print(f"\n\n  {'='*90}")
    print(f"  BATCH 5-7 LEADERBOARD (v53-v72)")
    print(f"  {'='*90}")
    ranked = sorted(all_results, key=lambda x: x.get("combined", 0) if not np.isnan(x.get("combined", 0)) else -999, reverse=True)
    for i, s in enumerate(ranked):
        tag = ""
        if s["name"] not in ("v49 ConsecPat (champion)", "v17 MFI (prev champ)"):
            if i == 0:
                tag = " ** NEW CHAMPION **"
            elif ranked[0]["name"] in ("v49 ConsecPat (champion)", "v17 MFI (prev champ)") and i == 1:
                if ranked[1]["name"] not in ("v49 ConsecPat (champion)", "v17 MFI (prev champ)"):
                    tag = " ** BEST NEW **"
        print(f"  #{i+1:2d}: {s['name']:<28} C={s['combined']:6.3f}  Sh={s['med_sh']:6.3f}  Spr={s['spread']:+5.1f}%  "
              f"GS={s.get('good_sep','?')}{tag}")

    # Update leaderboard
    try:
        with open(LEADERBOARD_FILE, "r") as f:
            existing = json.load(f)
    except Exception:
        existing = []
    existing_names = {e["name"] for e in existing}
    for s in all_results:
        if s["name"] not in ("v49 ConsecPat (champion)", "v17 MFI (prev champ)") and s["name"] not in existing_names:
            entry = {k: v for k, v in s.items() if not isinstance(v, float) or not np.isnan(v)}
            existing.append(entry)
    existing.sort(key=lambda x: x.get("combined", 0), reverse=True)
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"\n  Leaderboard updated ({len(existing)} entries)")
    print(f"  Total time: {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
