#!/usr/bin/env python3
"""
CSI BATCH 3-5: v33-v52 — Advanced Techniques + Radical Ideas
=============================================================
Goal: Beat v17 MFI (C=0.526, Spr=+1.2%)
Batch 3: v33-v42 (Advanced signal processing)
Batch 4: v43-v52 (Radical new approaches)
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
# BATCH 3: v33-v42 — Advanced Signal Processing
# ═══════════════════════════════════════════════════════════

def v33_entropy_regime(ind):
    """v33: Shannon entropy of return distribution as regime detector."""
    close = ind["close"]; ret = ind["ret_1"]

    # Bin returns into quintiles and compute entropy over rolling window
    def rolling_entropy(series, window=40, bins=10):
        vals = series.values.astype(float)
        out = np.full(len(vals), np.nan)
        for i in range(window, len(vals)):
            seg = vals[i-window:i]
            seg = seg[~np.isnan(seg)]
            if len(seg) < window // 2:
                continue
            counts, _ = np.histogram(seg, bins=bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            out[i] = -np.sum(probs * np.log2(probs))
        return pd.Series(out, index=series.index)

    entropy = rolling_entropy(ret, window=40, bins=10)
    max_ent = np.log2(10)  # Maximum entropy for 10 bins
    ent_norm = (entropy / max_ent).fillna(0.5)

    # High entropy = random/ranging market -> use mean reversion
    # Low entropy = trending/clustered -> use momentum
    is_trending = (1 - ent_norm).clip(0, 1)  # Low entropy = trending
    trend_sig = ind["trend_score"] * 0.5 + ind["mom_score"] * 0.5
    mr_sig = -(ind["rsi"] - 50) / 50 * 0.6 + -(ind["bb_pctb"] - 0.5) * 0.4

    raw = (is_trending * trend_sig + (1 - is_trending) * mr_sig) * 80
    raw = raw + ind["vol_flow"] * 15 + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v34_second_derivative(ind):
    """v34: Second derivative of trend (acceleration/deceleration detector)."""
    close = ind["close"]

    # First derivative: price momentum (EMA slope)
    ema20 = ind["ema_21"]
    slope1 = ema20.pct_change(5)  # First derivative

    # Second derivative: acceleration of momentum
    slope2 = slope1.diff(5)  # How fast momentum is changing
    slope2_n = (slope2 / slope2.abs().rolling(60, min_periods=20).max().replace(0, float("nan"))).clip(-1, 1)

    # Signal: positive acceleration = momentum building, negative = fading
    # Key insight: acceleration peaks BEFORE price peaks
    raw = (0.40 * slope2_n + 0.30 * ind["mom_score"] + 0.20 * ind["vol_flow"] + 0.10 * ind["ma_context"]) * 80

    # Early sell: momentum decelerating while price still high
    decel_sell = (slope2_n < -0.3).astype(float) * (ind["bb_pctb"] > 0.7).astype(float)
    raw = raw - decel_sell * 20

    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v35_vol_adjusted_mfi(ind):
    """v35: MFI enhanced with volatility context — widen thresholds in high vol."""
    close = ind["close"]; mfi = ind["mfi"]
    vol_pct = ind["vol_pct"].fillna(0.5)
    above_200 = ind["above_200"].fillna(0)

    # Volatility-adaptive MFI thresholds
    # In high vol: need more extreme MFI to signal (wider bands)
    # In low vol: tighter bands (subtle flows matter more)
    overbought = 60 + vol_pct * 20  # 60-80 range
    oversold = 40 - vol_pct * 20    # 20-40 range

    mfi_buy = pd.Series(np.where(mfi < oversold, (oversold - mfi) / oversold, 0.0), index=close.index).clip(0, 1)
    mfi_sell = pd.Series(np.where(mfi > overbought, (mfi - overbought) / (100 - overbought), 0.0), index=close.index).clip(0, 1)

    # Core MFI trend
    mfi_trend = (mfi - 50) / 50

    # Divergence (from v17)
    price_high = (close >= close.rolling(20).max() * 0.98).astype(float)
    mfi_weak = (mfi < mfi.rolling(20).max() * 0.90).astype(float)
    bearish_div = price_high * mfi_weak

    price_low = (close <= close.rolling(20).min() * 1.02).astype(float)
    mfi_strong = (mfi > mfi.rolling(20).min() * 1.10).astype(float)
    bullish_div = price_low * mfi_strong

    raw = mfi_trend * 40 + mfi_buy * 20 - mfi_sell * 20 + bullish_div * 20 - bearish_div * 20
    # Vol dampening
    raw = raw * ind["vol_dampener"]
    raw = raw + above_200 * 5
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v36_fractal_adaptive(ind):
    """v36: Fractal dimension (via Hurst) with improved regime adaptation."""
    close = ind["close"]; hurst = ind["hurst"].fillna(0.5)

    # Use Hurst to determine market character
    # H > 0.5 = trending (persistent), H < 0.5 = mean-reverting (anti-persistent)
    # H = 0.5 = random walk

    # Smoothed Hurst for regime stability
    h_smooth = hurst.ewm(span=10, adjust=False).mean().fillna(0.5)

    # Three regime signals
    trend_sig = ind["trend_score"] * 0.4 + ind["mom_score"] * 0.4 + ind["vol_flow"] * 0.2
    mr_sig = -(ind["rsi"] - 50) / 50 * 0.5 + -(ind["bb_pctb"] - 0.5) * 0.5
    random_sig = ind["vol_flow"] * 0.5 + ind["ma_context"] * 0.5  # When random, follow flow

    # Blend with soft boundaries
    trend_w = ((h_smooth - 0.55) / 0.15).clip(0, 1)  # Full trend above H=0.7
    mr_w = ((0.45 - h_smooth) / 0.15).clip(0, 1)     # Full MR below H=0.3
    random_w = 1 - trend_w - mr_w
    random_w = random_w.clip(0, 1)

    raw = (trend_w * trend_sig + mr_w * mr_sig + random_w * random_sig) * 80
    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v37_multi_tf_consensus(ind):
    """v37: Multi-timeframe consensus with confidence weighting."""
    close = ind["close"]

    # Short-term signal (5-10 day)
    st = 0.5 * ind["mom_5"] + 0.3 * ind["macd_n"] + 0.2 * ind["vol_flow"]
    # Medium-term signal (10-20 day)
    mt = 0.5 * ind["mom_10"] + 0.3 * ind["trend_score"] + 0.2 * ind["ma_context"]
    # Long-term signal (20-40 day)
    lt = 0.5 * ind["mom_20"] + 0.3 * ind["above_50"].astype(float) * 2 - 0.3 + 0.2 * ind["mom_40"]

    # Agreement detection
    signs = pd.concat([np.sign(st), np.sign(mt), np.sign(lt)], axis=1)
    agreement = signs.sum(axis=1).abs() / 3  # 0.33 = disagree, 1.0 = all agree

    # Confidence: high when all timeframes align
    confidence = agreement ** 2  # Penalize disagreement quadratically

    # Weighted average by timeframe importance
    raw = (0.30 * st + 0.40 * mt + 0.30 * lt) * confidence * 100
    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v38_adaptive_weights(ind):
    """v38: Dynamically reweight components based on recent correlation with returns."""
    close = ind["close"]; ret = ind["ret_1"]

    # Component signals
    sig_mom = ind["mom_score"]
    sig_trend = ind["trend_score"]
    sig_vol = ind["vol_flow"]
    sig_mr = -(ind["rsi"] - 50) / 50
    sig_mfi = (ind["mfi"] - 50) / 50

    # Rolling correlation of each signal with future 5d returns
    fwd = close.pct_change(5).shift(-5)
    window = 60

    def roll_corr(sig):
        return sig.rolling(window, min_periods=30).corr(fwd.shift(5)).fillna(0).clip(-0.5, 0.5)

    w_mom = roll_corr(sig_mom).abs()
    w_trend = roll_corr(sig_trend).abs()
    w_vol = roll_corr(sig_vol).abs()
    w_mr = roll_corr(sig_mr).abs()
    w_mfi = roll_corr(sig_mfi).abs()

    # Normalize weights
    w_total = (w_mom + w_trend + w_vol + w_mr + w_mfi).replace(0, 1)
    raw = ((w_mom * sig_mom + w_trend * sig_trend + w_vol * sig_vol +
            w_mr * sig_mr + w_mfi * sig_mfi) / w_total * 80)

    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v39_vwap_deviation(ind):
    """v39: Volume-Weighted Average Price deviation (VWAP proxy from daily data)."""
    close = ind["close"]; volume = ind["volume"]
    high = ind["high"]; low = ind["low"]

    # VWAP proxy: volume-weighted typical price
    tp = (high + low + close) / 3
    vwap_20 = (tp * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, float("nan"))

    # Deviation from VWAP
    dev = ((close - vwap_20) / vwap_20 * 100).clip(-10, 10)
    dev_n = (dev / dev.abs().rolling(60, min_periods=20).max().replace(0, float("nan"))).clip(-1, 1)

    # VWAP trend (is VWAP rising or falling?)
    vwap_slope = vwap_20.pct_change(5)
    vwap_slope_n = (vwap_slope / vwap_slope.abs().rolling(60, min_periods=20).max().replace(0, float("nan"))).clip(-1, 1)

    raw = (0.35 * dev_n + 0.30 * vwap_slope_n + 0.20 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 80
    raw = raw + ind["above_200"].fillna(0) * 5
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v40_relative_range(ind):
    """v40: Position in recent range with flow confirmation."""
    close = ind["close"]
    donch = ind["donch_pct"]  # 0-1 position in 20d range

    # Extended range lookbacks
    h60 = ind["high"].rolling(60).max()
    l60 = ind["low"].rolling(60).min()
    range_60 = ((close - l60) / (h60 - l60).replace(0, float("nan"))).clip(0, 1)

    h120 = ind["high"].rolling(120, min_periods=60).max()
    l120 = ind["low"].rolling(120, min_periods=60).min()
    range_120 = ((close - l120) / (h120 - l120).replace(0, float("nan"))).clip(0, 1)

    # Signal: where in range + direction of movement within range
    pos = (0.4 * donch + 0.35 * range_60 + 0.25 * range_120)
    pos_centered = (pos - 0.5) * 2  # -1 to +1

    # Momentum within range (breakout tendency)
    range_mom = pos.diff(5)
    range_mom_n = (range_mom / range_mom.abs().rolling(30, min_periods=10).max().replace(0, float("nan"))).clip(-1, 1)

    raw = (0.35 * pos_centered + 0.30 * range_mom_n + 0.20 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 80
    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v41_mfi_regime_corrections(ind):
    """v41: MFI core + regime-specific corrections (combine v17's strength with smart corrections)."""
    close = ind["close"]; mfi = ind["mfi"]; rsi = ind["rsi"]
    above_200 = ind["above_200"].fillna(0)
    vol_pct = ind["vol_pct"].fillna(0.5)
    adx_r = ind["adx_regime"].fillna(0.5)

    # Core MFI from v17
    mfi_sig = (mfi - 50) / 50
    price_high = (close >= close.rolling(20).max() * 0.98).astype(float)
    mfi_weak = (mfi < mfi.rolling(20).max() * 0.90).astype(float)
    bearish_div = price_high * mfi_weak
    price_low = (close <= close.rolling(20).min() * 1.02).astype(float)
    mfi_strong = (mfi > mfi.rolling(20).min() * 1.10).astype(float)
    bullish_div = price_low * mfi_strong

    raw = mfi_sig * 50 + bullish_div * 25 - bearish_div * 25

    # Regime-specific corrections
    # 1. In strong trends (high ADX), trust MFI trend direction more
    raw = raw * (0.7 + 0.3 * adx_r)

    # 2. In high vol, dampen signals (wider noise bands)
    raw = raw * (1.1 - 0.2 * vol_pct)

    # 3. Oversold in uptrend correction (from v8 — proven to work)
    oversold = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    raw = raw + oversold * above_200 * 30

    # 4. Overbought exhaustion when MA50 rolling over
    rsi_ob = pd.Series(np.where(rsi > 70, (rsi - 70) / 30, 0.0), index=close.index).clip(0, 1)
    raw = raw - rsi_ob * (ind["ma50_slope"] < 0).astype(float) * 15

    raw = raw + above_200 * 3 + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v42_information_blend(ind):
    """v42: Blend signals by information content (high IC signals get more weight)."""
    close = ind["close"]

    # Signals
    sigs = {
        "mom": ind["mom_score"],
        "trend": ind["trend_score"],
        "vol_flow": ind["vol_flow"],
        "mfi": (ind["mfi"] - 50) / 50,
        "rsi_mr": -(ind["rsi"] - 50) / 50,
        "struct": (ind["hl_ratio"] / 5).clip(-1, 1),
        "donch": (ind["donch_pct"] - 0.5) * 2,
        "bb": -(ind["bb_pctb"] - 0.5) * 2,
    }

    # Equal weight baseline (avoid lookahead)
    raw = sum(v for v in sigs.values()) / len(sigs) * 80

    # But add MFI divergence (proven powerful)
    mfi = ind["mfi"]
    price_high = (close >= close.rolling(20).max() * 0.98).astype(float)
    mfi_weak = (mfi < mfi.rolling(20).max() * 0.90).astype(float)
    raw = raw - price_high * mfi_weak * 20

    price_low = (close <= close.rolling(20).min() * 1.02).astype(float)
    mfi_strong = (mfi > mfi.rolling(20).min() * 1.10).astype(float)
    raw = raw + price_low * mfi_strong * 20

    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


# ═══════════════════════════════════════════════════════════
# BATCH 4: v43-v52 — Radical New Approaches
# ═══════════════════════════════════════════════════════════

def v43_pure_volume(ind):
    """v43: Pure volume microstructure (ignore price trend, only volume patterns)."""
    close = ind["close"]; volume = ind["volume"]

    # Volume relative to average
    vol_rel = ind["vol_rel"]

    # Volume trend (increasing or decreasing)
    vol_sma5 = volume.rolling(5).mean()
    vol_sma20 = volume.rolling(20).mean()
    vol_trend = ((vol_sma5 - vol_sma20) / vol_sma20.replace(0, float("nan"))).clip(-1, 1)

    # Volume-price relationship
    # High volume + up move = accumulation
    # High volume + down move = distribution
    up_vol = volume.where(close > close.shift(1), 0.0)
    dn_vol = volume.where(close <= close.shift(1), 0.0)
    accum = up_vol.rolling(10).sum() / volume.rolling(10).sum().replace(0, float("nan"))
    accum_n = (accum - 0.5) * 2  # -1 to +1

    # OBV momentum
    obv_sig = ind["obv_signal"]

    # A/D line
    ad_sig = ind["ad_score"]

    raw = (0.30 * accum_n + 0.25 * obv_sig + 0.25 * ad_sig + 0.20 * ind["vol_ratio"]) * 80
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100).ewm(span=4, adjust=False).mean()


def v44_return_shape(ind):
    """v44: Return distribution shape — skewness and kurtosis as regime signals."""
    close = ind["close"]; ret = ind["ret_1"]

    # Rolling skewness (negative = left tail, positive = right tail)
    skew = ret.rolling(30, min_periods=15).skew().fillna(0).clip(-3, 3)
    # Rolling kurtosis (high = fat tails, low = thin tails)
    kurt = ret.rolling(30, min_periods=15).apply(
        lambda x: pd.Series(x).kurtosis() if len(x) > 5 else 0, raw=True
    ).fillna(0).clip(-5, 10)

    # Skewness signal: positive skew = more upside potential
    skew_n = (skew / 2).clip(-1, 1)

    # Kurtosis as risk: high kurtosis = tail risk environment
    kurt_risk = ((kurt - 3) / 5).clip(0, 1)  # Excess kurtosis
    dampener = 1.0 - 0.3 * kurt_risk

    raw = (0.35 * skew_n + 0.30 * ind["trend_score"] + 0.20 * ind["vol_flow"] + 0.15 * ind["mom_score"]) * 80 * dampener
    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v45_momentum_crash_avoid(ind):
    """v45: Momentum with crash avoidance — detect when momentum is about to reverse."""
    close = ind["close"]; rsi = ind["rsi"]

    # Base momentum
    mom = ind["mom_score"]

    # Crash warning signals:
    # 1. RSI divergence (price new high, RSI lower high)
    price_20h = close.rolling(20).max()
    at_high = (close >= price_20h * 0.98).astype(float)
    rsi_20h = rsi.rolling(20).max()
    rsi_lower = (rsi < rsi_20h * 0.90).astype(float)
    bearish_rsi_div = at_high * rsi_lower

    # 2. Volume divergence at highs
    vol_10 = ind["volume"].rolling(10).mean()
    vol_30 = ind["volume"].rolling(30).mean()
    vol_declining = (vol_10 < vol_30 * 0.85).astype(float)
    vol_div_at_high = at_high * vol_declining

    # 3. Momentum deceleration (v34 insight)
    mom_accel = ind["mom_accel_n"]
    decel_at_high = (mom_accel < -0.3).astype(float) * (ind["bb_pctb"] > 0.7).astype(float)

    # Crash probability estimate
    crash_prob = (0.40 * bearish_rsi_div + 0.30 * vol_div_at_high + 0.30 * decel_at_high).clip(0, 1)

    raw = mom * 70 * (1 - 0.6 * crash_prob)  # Reduce signal when crash likely
    raw = raw - crash_prob * 25  # Add negative bias during crash warning

    # Oversold recovery (from v8)
    above_200 = ind["above_200"].fillna(0)
    oversold = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    raw = raw + oversold * above_200 * 30

    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v46_spy_conditioned(ind):
    """v46: Use SPY regime to condition individual stock signals.
    Note: This requires pre-computing SPY indicators."""
    close = ind["close"]

    # Get SPY indicators (if available)
    spy_ind = get_indicators("SPY")
    if spy_ind is None:
        return v17_mfi_divergence(ind)  # Fallback

    # SPY regime
    spy_trend = spy_ind["trend_score"].reindex(close.index, method="ffill").fillna(0)
    spy_vol_pct = spy_ind["vol_pct"].reindex(close.index, method="ffill").fillna(0.5)
    spy_above_200 = spy_ind["above_200"].reindex(close.index, method="ffill").fillna(0.5)

    # Base signal (MFI from v17)
    base = v17_mfi_divergence(ind)

    # Market regime adjustments
    # In bull market: amplify buy signals
    bull_boost = (spy_above_200 > 0.5).astype(float) * (spy_trend > 0).astype(float)
    # In bear market: amplify sell signals
    bear_boost = (spy_above_200 < 0.5).astype(float) * (spy_trend < 0).astype(float)

    result = base + bull_boost * base.clip(0, None) * 0.2 - bear_boost * (-base).clip(0, None) * 0.2

    # High market vol = reduce all signals
    vol_adj = 1.0 - 0.2 * (spy_vol_pct - 0.5).clip(0, 0.5)
    result = result * vol_adj
    return result.clip(-100, 100)


def v47_trend_speed(ind):
    """v47: Detect trending vs ranging FAST using price efficiency + ADX combo."""
    close = ind["close"]
    eff = ind["efficiency"].fillna(0.5)
    adx_r = ind["adx_regime"].fillna(0.5)

    # Combined regime score (0 = choppy, 1 = strong trend)
    regime = (0.5 * eff + 0.5 * adx_r).clip(0, 1)

    # In trending: follow momentum aggressively
    trend_sig = (ind["mom_score"] * 0.5 + ind["trend_score"] * 0.5) * 90

    # In ranging: mean revert with volume confirmation
    mr_sig = (-(ind["rsi"] - 50) / 50 * 0.4 + -(ind["bb_pctb"] - 0.5) * 0.3 + ind["vol_flow"] * 0.3) * 60

    raw = regime * trend_sig + (1 - regime) * mr_sig

    # MFI sell overlay (proven effective)
    mfi = ind["mfi"]
    price_high = (close >= close.rolling(20).max() * 0.98).astype(float)
    mfi_weak = (mfi < mfi.rolling(20).max() * 0.90).astype(float)
    raw = raw - price_high * mfi_weak * 15

    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v48_fib_reversion(ind):
    """v48: Fibonacci retracement levels as mean-reversion targets."""
    close = ind["close"]; high = ind["high"]; low = ind["low"]

    # Recent swing: 40-day high/low
    h40 = high.rolling(40).max()
    l40 = low.rolling(40).min()
    swing_range = (h40 - l40).replace(0, float("nan"))

    # Fibonacci levels
    fib_382 = h40 - swing_range * 0.382
    fib_500 = h40 - swing_range * 0.500
    fib_618 = h40 - swing_range * 0.618

    # Distance to nearest Fibonacci level (proxy for mean-reversion potential)
    dist_382 = (close - fib_382) / swing_range
    dist_500 = (close - fib_500) / swing_range
    dist_618 = (close - fib_618) / swing_range

    # Buy near support (close to 0.618), sell near resistance (close to top)
    near_support = (dist_618.abs() < 0.05).astype(float)  # Near 61.8% retracement
    near_mid = (dist_500.abs() < 0.05).astype(float)      # Near 50% retracement
    near_resist = (dist_382.abs() < 0.05).astype(float)   # Near 38.2% (resistance)

    # Position in range as continuous signal
    range_pos = ((close - l40) / swing_range).clip(0, 1)
    range_sig = -(range_pos - 0.5) * 2  # Buy low, sell high

    raw = (0.40 * range_sig + 0.25 * ind["trend_score"] + 0.20 * ind["vol_flow"] + 0.15 * ind["mom_score"]) * 70
    # Fib level bonuses
    raw = raw + near_support * (ind["above_200"].fillna(0)) * 15
    raw = raw - near_resist * (ind["ma50_slope"] < 0).astype(float) * 10
    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=4, adjust=False).mean()


def v49_consecutive_patterns(ind):
    """v49: Consecutive candle patterns (runs of up/down days)."""
    close = ind["close"]

    # Consecutive up/down days
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)

    # Count consecutive runs
    def count_runs(series):
        """Count consecutive 1s, reset on 0."""
        vals = series.values.astype(float)
        out = np.zeros(len(vals))
        for i in range(1, len(vals)):
            if vals[i] > 0:
                out[i] = out[i-1] + 1
            else:
                out[i] = 0
        return pd.Series(out, index=series.index)

    up_runs = count_runs(up)
    dn_runs = count_runs(dn)

    # Mean reversion: long runs tend to reverse
    # 3+ consecutive days = exhaustion likely
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)

    # Momentum: 2-3 day runs = continuation
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)

    mr_sig = dn_exhaust * 0.5 - up_exhaust * 0.5  # Reverse after long runs
    mom_sig = up_mom * 0.3 - dn_mom * 0.3  # Continue short runs

    raw = (mr_sig + mom_sig + 0.30 * ind["vol_flow"] + 0.20 * ind["trend_score"]) * 80
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v50_order_flow_proxy(ind):
    """v50: Order flow imbalance proxy from OHLC bar structure."""
    close = ind["close"]; high = ind["high"]; low = ind["low"]
    op = close.shift(1)  # Use previous close as open proxy

    # Bar structure metrics
    # Upper shadow = selling pressure, lower shadow = buying pressure
    body = close - op
    upper_shadow = high - np.maximum(close, op)
    lower_shadow = np.minimum(close, op) - low
    full_range = (high - low).replace(0, float("nan"))

    # Body ratio: how much of the bar is body vs shadows
    body_ratio = (body.abs() / full_range).clip(0, 1)

    # Buying pressure: large lower shadows = buyers stepped in
    buy_pressure = (lower_shadow / full_range).clip(0, 1)
    # Selling pressure: large upper shadows = sellers stepped in
    sell_pressure = (upper_shadow / full_range).clip(0, 1)

    # Net order flow proxy
    net_flow = (buy_pressure - sell_pressure).rolling(5).mean()
    net_flow_n = (net_flow / net_flow.abs().rolling(30, min_periods=10).max().replace(0, float("nan"))).clip(-1, 1)

    # Body direction with confidence (large body = strong conviction)
    body_sig = (body / full_range).clip(-1, 1).rolling(5).mean()

    raw = (0.35 * net_flow_n + 0.25 * body_sig + 0.25 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 80
    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v51_mfi_with_everything(ind):
    """v51: MFI core + every proven enhancement (kitchen sink approach)."""
    close = ind["close"]; mfi = ind["mfi"]; rsi = ind["rsi"]
    above_200 = ind["above_200"].fillna(0)
    vol_pct = ind["vol_pct"].fillna(0.5)

    # Core MFI (from v17)
    mfi_sig = (mfi - 50) / 50
    price_high = (close >= close.rolling(20).max() * 0.98).astype(float)
    mfi_weak = (mfi < mfi.rolling(20).max() * 0.90).astype(float)
    bearish_div = price_high * mfi_weak
    price_low = (close <= close.rolling(20).min() * 1.02).astype(float)
    mfi_strong = (mfi > mfi.rolling(20).min() * 1.10).astype(float)
    bullish_div = price_low * mfi_strong

    raw = mfi_sig * 45 + bullish_div * 20 - bearish_div * 20

    # Enhancement 1: Second derivative crash detection (from v34/v45)
    mom_accel = ind["mom_accel_n"]
    decel_sell = (mom_accel < -0.3).astype(float) * (ind["bb_pctb"] > 0.7).astype(float)
    raw = raw - decel_sell * 12

    # Enhancement 2: Structure confirmation (from v22)
    struct_confirm = (np.sign(raw) == np.sign(ind["hl_ratio"])).astype(float)
    raw = raw * (0.7 + 0.3 * struct_confirm)

    # Enhancement 3: Volume flow confirmation
    vol_confirm = (np.sign(raw) * np.sign(ind["vol_flow"])).clip(0, 1)
    raw = raw * (0.8 + 0.2 * vol_confirm)

    # Enhancement 4: Oversold in uptrend (from v8)
    oversold = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    raw = raw + oversold * above_200 * 25

    # Enhancement 5: Vol dampening
    raw = raw * (1.1 - 0.2 * vol_pct)

    raw = raw + above_200 * 3 + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v52_mfi_asymmetric(ind):
    """v52: MFI with asymmetric processing — different logic for buy vs sell signals."""
    close = ind["close"]; mfi = ind["mfi"]; rsi = ind["rsi"]
    above_200 = ind["above_200"].fillna(0)

    # BUY SIDE: MFI bullish divergence + trend confirmation + oversold
    mfi_buy_trend = pd.Series(np.where(mfi > 50, (mfi - 50) / 50, 0.0), index=close.index)
    price_low = (close <= close.rolling(20).min() * 1.02).astype(float)
    mfi_strong = (mfi > mfi.rolling(20).min() * 1.10).astype(float)
    bullish_div = price_low * mfi_strong
    oversold = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)

    buy_sig = (0.35 * mfi_buy_trend + 0.25 * bullish_div + 0.20 * ind["trend_score"].clip(0, None) +
               0.10 * ind["vol_flow"].clip(0, None) + 0.10 * oversold * above_200)

    # SELL SIDE: MFI bearish divergence + exhaustion + volume decline
    mfi_sell_trend = pd.Series(np.where(mfi < 50, (50 - mfi) / 50, 0.0), index=close.index)
    price_high = (close >= close.rolling(20).max() * 0.98).astype(float)
    mfi_weak = (mfi < mfi.rolling(20).max() * 0.90).astype(float)
    bearish_div = price_high * mfi_weak
    rsi_ob = pd.Series(np.where(rsi > 68, (rsi - 68) / 32, 0.0), index=close.index).clip(0, 1)
    mom_decel = (-ind["mom_accel_n"]).clip(0, 1)

    sell_sig = (0.30 * mfi_sell_trend + 0.25 * bearish_div + 0.20 * rsi_ob +
                0.15 * mom_decel + 0.10 * (-ind["vol_flow"]).clip(0, None))

    raw = (buy_sig - sell_sig) * 100
    raw = raw + above_200 * 3 + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def main():
    print("=" * 110)
    print("  CSI BATCH 3-4: v33-v52 — Advanced Techniques + Radical Ideas")
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
        ("v17 MFI (champion)", v17_mfi_divergence),
        # Batch 3: Advanced
        ("v33 Entropy Regime", v33_entropy_regime),
        ("v34 2nd Derivative", v34_second_derivative),
        ("v35 Vol-Adj MFI", v35_vol_adjusted_mfi),
        ("v36 Fractal Adaptive", v36_fractal_adaptive),
        ("v37 Multi-TF Consensus", v37_multi_tf_consensus),
        ("v38 Adaptive Weights", v38_adaptive_weights),
        ("v39 VWAP Deviation", v39_vwap_deviation),
        ("v40 Relative Range", v40_relative_range),
        ("v41 MFI+Regime Corr", v41_mfi_regime_corrections),
        ("v42 Information Blend", v42_information_blend),
        # Batch 4: Radical
        ("v43 Pure Volume", v43_pure_volume),
        ("v44 Return Shape", v44_return_shape),
        ("v45 Mom Crash Avoid", v45_momentum_crash_avoid),
        ("v46 SPY Conditioned", v46_spy_conditioned),
        ("v47 Trend Speed", v47_trend_speed),
        ("v48 Fib Reversion", v48_fib_reversion),
        ("v49 Consec Patterns", v49_consecutive_patterns),
        ("v50 Order Flow Proxy", v50_order_flow_proxy),
        ("v51 MFI+Everything", v51_mfi_with_everything),
        ("v52 MFI Asymmetric", v52_mfi_asymmetric),
    ]

    all_results = []
    for name, fn in strategies:
        t = time.time()
        s = run_strategy(name, fn)
        s["time"] = round(time.time() - t, 1)
        all_results.append(s)

    print(f"\n\n  {'='*80}")
    print(f"  BATCH 3-4 LEADERBOARD (v33-v52)")
    print(f"  {'='*80}")
    ranked = sorted(all_results, key=lambda x: x.get("combined", 0) if not np.isnan(x.get("combined", 0)) else -999, reverse=True)
    for i, s in enumerate(ranked):
        tag = ""
        if s["name"] != "v17 MFI (champion)":
            if i == 0:
                tag = " ** NEW CHAMPION **"
            elif i == 1 and ranked[0]["name"] == "v17 MFI (champion)":
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
        if s["name"] not in ("v17 MFI (champion)",) and s["name"] not in existing_names:
            entry = {k: v for k, v in s.items() if not isinstance(v, float) or not np.isnan(v)}
            existing.append(entry)
    existing.sort(key=lambda x: x.get("combined", 0), reverse=True)
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"\n  Leaderboard updated ({len(existing)} entries)")
    print(f"  Total time: {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
