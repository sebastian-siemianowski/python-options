#!/usr/bin/env python3
"""
CSI BATCH 2: v23-v32 — Hybrid combinations of Batch 1 winners
==============================================================
Top performers from Batch 1:
  v17 MFI Divergence   C=0.526  Spr=+1.2%  SellC=35/98  (BEST sell detector)
  v22 Structure Based  C=0.507  Sh=0.511   BuyP=87/98   (BEST buy detector)
  v14 Mean Reversion   C=0.399  Spr=+4.2%  GS=59/98     (BEST regime separation)
  v16 Vol Breakout     C=0.493  Sh=0.499   GS=45/98     (BEST balance)
"""

import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

# Import everything from mega harness
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi_mega_harness import (
    UNIVERSE, preload_all, get_indicators, evaluate, summarize, run_strategy,
    v8_baseline, v13_kalman_trend, v14_mean_reversion, v15_momentum_persistence,
    v16_volatility_breakout, v17_mfi_divergence, v18_cci_williams, v19_ichimoku_trend,
    v20_hurst_adaptive, v22_structure_based, LEADERBOARD_FILE,
)


def v23_mfi_structure(ind):
    """v23: MFI sell detection + Structure buy detection hybrid."""
    close = ind["close"]; mfi = ind["mfi"]; rsi = ind["rsi"]
    hl = ind["hl_ratio"]; donch = ind["donch_pct"]
    above_200 = ind["above_200"].fillna(0)

    # BUY side: Structure-based (from v22)
    struct_n = (hl / 5).clip(-1, 1)
    donch_sig = (donch - 0.5) * 2
    buy_sig = (0.35 * struct_n + 0.25 * donch_sig + 0.25 * ind["trend_score"] + 0.15 * ind["vol_flow"]).clip(0, None)

    # SELL side: MFI-based (from v17)
    mfi_sig = -(mfi - 50) / 50  # MFI < 50 = negative flow
    mfi_sig = mfi_sig.clip(0, None)  # Only sell side
    price_high_20 = (close >= close.rolling(20).max() * 0.98).astype(float)
    mfi_not_high = (mfi < mfi.rolling(20).max() * 0.90).astype(float)
    bearish_div = price_high_20 * mfi_not_high
    sell_sig = 0.6 * mfi_sig + 0.4 * bearish_div

    raw = (buy_sig - sell_sig) * 90
    # v8 oversold correction
    oversold = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    raw = raw + oversold * above_200 * 35 - oversold * (1 - above_200) * 8
    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v24_mfi_v8_corrections(ind):
    """v24: v17 MFI base + all v8 corrections (the sell powerhouse + correction engine)."""
    close = ind["close"]; mfi = ind["mfi"]; rsi = ind["rsi"]
    above_200 = ind["above_200"].fillna(0)

    # Base from v17
    mfi_sig = (mfi - 50) / 50
    price_high_20 = (close >= close.rolling(20).max() * 0.98).astype(float)
    mfi_not_high = (mfi < mfi.rolling(20).max() * 0.90).astype(float)
    bearish_div = price_high_20 * mfi_not_high
    price_low_20 = (close <= close.rolling(20).min() * 1.02).astype(float)
    mfi_not_low = (mfi > mfi.rolling(20).min() * 1.10).astype(float)
    bullish_div = price_low_20 * mfi_not_low

    raw = mfi_sig * 50 + bullish_div * 25 - bearish_div * 25

    # Full v8 correction suite
    stoch_k = ind["stoch_k"]
    oversold_c = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    stoch_os = pd.Series(np.where(stoch_k < 20, (20 - stoch_k) / 20, 0.0), index=close.index).clip(0, 1)
    os_strength = np.maximum(oversold_c, stoch_os)
    raw = raw + os_strength * above_200 * 40 - os_strength * (1 - above_200) * 10

    rsi_ob = pd.Series(np.where(rsi > 68, (rsi - 68) / 32, 0.0), index=close.index).clip(0, 1)
    raw = raw - rsi_ob * (ind["ma50_slope"] < 0).astype(float) * 25
    rsi_max10 = rsi.rolling(10).max()
    raw = raw - (rsi_max10 > 70).astype(float) * ((rsi_max10 - rsi) / 20).clip(0, 1) * 12

    price_near_high = ((close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min()).replace(0, float("nan"))).clip(0, 1)
    vol_10 = ind["volume"].rolling(10).mean()
    vol_30 = ind["volume"].rolling(30).mean()
    raw = raw - (price_near_high > 0.70).astype(float) * ((vol_30 - vol_10) / vol_30.replace(0, float("nan"))).clip(0, 1) * 8
    raw = raw - (1 - ind["above_50"]).astype(float) * (vol_10 > vol_30 * 1.1).astype(float) * (raw < 0).astype(float) * 8

    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v25_mr_mfi(ind):
    """v25: Mean Reversion regime separation + MFI sell detection."""
    close = ind["close"]; rsi = ind["rsi"]; mfi = ind["mfi"]
    bb = ind["bb_pctb"]; above_200 = ind["above_200"].fillna(0)

    # Mean reversion core (regime separation from v14)
    mr_rsi = -(rsi - 50) / 50
    mr_bb = -(bb - 0.5) * 2
    mr_stoch = -(ind["stoch_k"] - 50) / 50
    mr_base = 0.40 * mr_rsi + 0.35 * mr_bb + 0.25 * mr_stoch

    # MFI enhancement (sell detection from v17)
    mfi_sig = (mfi - 50) / 50
    price_high = (close >= close.rolling(20).max() * 0.98).astype(float)
    mfi_weak = (mfi < mfi.rolling(20).max() * 0.90).astype(float)
    bearish_div = price_high * mfi_weak

    raw = mr_base * 60 + mfi_sig * 20 - bearish_div * 20
    raw = raw * (0.5 + 0.5 * above_200)
    raw = np.where((raw > 0) & (above_200 < 0.5), raw * 0.4, raw)
    raw = pd.Series(raw, index=close.index) + 1
    return raw.clip(-100, 100).ewm(span=4, adjust=False).mean()


def v26_structure_breakout(ind):
    """v26: Structure (v22) + Volatility Breakout (v16) fusion."""
    close = ind["close"]
    hl = ind["hl_ratio"]; donch = ind["donch_pct"]
    squeeze = ind["bb_squeeze"]; bb = ind["bb_pctb"]
    above_200 = ind["above_200"].fillna(0)

    # Structure component
    struct_n = (hl / 5).clip(-1, 1)
    donch_sig = (donch - 0.5) * 2

    # Breakout component
    is_squeezed = (squeeze < 0.2).astype(float)
    was_squeezed = is_squeezed.rolling(5).max()
    breakout_up = (bb > 1.0).astype(float) * was_squeezed
    breakout_dn = (bb < 0.0).astype(float) * was_squeezed
    breakout_sig = breakout_up - breakout_dn

    raw = (0.30 * struct_n + 0.20 * breakout_sig + 0.25 * ind["trend_score"] +
           0.15 * ind["vol_flow"] + 0.10 * donch_sig) * 85
    raw = raw + above_200 * 3
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v27_top4_ensemble(ind):
    """v27: Equal-weight ensemble of top 4 from Batch 1."""
    s1 = v17_mfi_divergence(ind)
    s2 = v22_structure_based(ind)
    s3 = v16_volatility_breakout(ind)
    s4 = v14_mean_reversion(ind)
    return ((s1 + s2 + s3 + s4) / 4).clip(-100, 100)


def v28_weighted_ensemble(ind):
    """v28: Combined-score-weighted ensemble of top 4."""
    s1 = v17_mfi_divergence(ind)   # C=0.526
    s2 = v22_structure_based(ind)   # C=0.507
    s3 = v16_volatility_breakout(ind)  # C=0.493
    s4 = v14_mean_reversion(ind)    # C=0.399
    total = 0.526 + 0.507 + 0.493 + 0.399
    w1, w2, w3, w4 = 0.526/total, 0.507/total, 0.493/total, 0.399/total
    return (w1*s1 + w2*s2 + w3*s3 + w4*s4).clip(-100, 100)


def v29_mfi_enhanced(ind):
    """v29: Enhanced MFI with tighter divergence + multi-TF confirmation."""
    close = ind["close"]; mfi = ind["mfi"]; rsi = ind["rsi"]
    above_200 = ind["above_200"].fillna(0)

    # Core MFI
    mfi_sig = (mfi - 50) / 50

    # Tighter divergence detection (5-day, 10-day, 20-day windows)
    div_score = pd.Series(0.0, index=close.index)
    for window in [10, 20, 30]:
        p_high = (close >= close.rolling(window).max() * 0.97).astype(float)
        m_weak = (mfi < mfi.rolling(window).max() * 0.85).astype(float)
        p_low = (close <= close.rolling(window).min() * 1.03).astype(float)
        m_strong = (mfi > mfi.rolling(window).min() * 1.15).astype(float)
        div_score = div_score + p_low * m_strong * 0.33 - p_high * m_weak * 0.33

    # Multi-TF MFI: also compute 7-period and 28-period MFI
    tp = (ind["high"] + ind["low"] + close) / 3
    mf_raw = tp * ind["volume"]
    for period in [7, 28]:
        mf_pos = mf_raw.where(tp > tp.shift(1), 0.0).rolling(period).sum()
        mf_neg = mf_raw.where(tp <= tp.shift(1), 0.0).rolling(period).sum()
        mf_r = mf_pos / mf_neg.replace(0, float("nan"))
        mfi_p = 100 - (100 / (1 + mf_r))
        mfi_sig = mfi_sig + (mfi_p - 50) / 50 * 0.3

    raw = mfi_sig * 45 + div_score * 30 + ind["trend_score"] * 15 + ind["vol_flow"] * 10
    raw = raw + above_200 * 4
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v30_regime_switch(ind):
    """v30: Regime-based strategy switching — MR in range, MFI+Trend in trends."""
    close = ind["close"]
    adx_r = ind["adx_regime"].fillna(0.5)

    # Trending regime: MFI + Trend
    trend_csi = v17_mfi_divergence(ind)
    # Ranging regime: Mean Reversion
    mr_csi = v14_mean_reversion(ind)

    # ADX-based blending
    result = adx_r * trend_csi + (1 - adx_r) * mr_csi
    return result.clip(-100, 100)


def v31_structure_v8corr(ind):
    """v31: Structure base (v22) + full v8 correction suite."""
    close = ind["close"]; rsi = ind["rsi"]
    above_200 = ind["above_200"].fillna(0)
    hl = ind["hl_ratio"]; donch = ind["donch_pct"]

    # Structure core from v22
    struct_n = (hl / 5).clip(-1, 1)
    donch_sig = (donch - 0.5) * 2
    raw = (0.35 * struct_n + 0.25 * donch_sig + 0.25 * ind["trend_score"] + 0.15 * ind["vol_flow"]) * 80

    # Full v8 corrections
    stoch_k = ind["stoch_k"]
    oversold_c = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    stoch_os = pd.Series(np.where(stoch_k < 20, (20 - stoch_k) / 20, 0.0), index=close.index).clip(0, 1)
    os_strength = np.maximum(oversold_c, stoch_os)
    raw = raw + os_strength * above_200 * 40 - os_strength * (1 - above_200) * 10

    rsi_ob = pd.Series(np.where(rsi > 68, (rsi - 68) / 32, 0.0), index=close.index).clip(0, 1)
    raw = raw - rsi_ob * (ind["ma50_slope"] < 0).astype(float) * 25
    rsi_max10 = rsi.rolling(10).max()
    raw = raw - (rsi_max10 > 70).astype(float) * ((rsi_max10 - rsi) / 20).clip(0, 1) * 12

    price_near_high = ((close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min()).replace(0, float("nan"))).clip(0, 1)
    vol_10 = ind["volume"].rolling(10).mean()
    vol_30 = ind["volume"].rolling(30).mean()
    raw = raw - (price_near_high > 0.70).astype(float) * ((vol_30 - vol_10) / vol_30.replace(0, float("nan"))).clip(0, 1) * 8
    raw = raw - (1 - ind["above_50"]).astype(float) * (vol_10 > vol_30 * 1.1).astype(float) * (raw < 0).astype(float) * 8

    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v32_sell_specialist(ind):
    """v32: MFI sell signals + Structure buy signals — specialist hybrid."""
    close = ind["close"]; mfi = ind["mfi"]; rsi = ind["rsi"]
    hl = ind["hl_ratio"]; donch = ind["donch_pct"]
    above_200 = ind["above_200"].fillna(0)

    # Buy side: Structure (v22 approach)
    struct_n = (hl / 5).clip(-1, 1)
    donch_sig = (donch - 0.5) * 2
    buy_str = (0.35 * struct_n.clip(0, None) + 0.25 * donch_sig.clip(0, None) +
               0.25 * ind["trend_score"].clip(0, None) + 0.15 * ind["vol_flow"].clip(0, None))

    # Sell side: MFI (v17 approach)
    mfi_sell = pd.Series(np.where(mfi < 45, (45 - mfi) / 45, 0.0), index=close.index).clip(0, 1)
    price_high = (close >= close.rolling(20).max() * 0.98).astype(float)
    mfi_weak = (mfi < mfi.rolling(20).max() * 0.90).astype(float)
    bearish_div = price_high * mfi_weak
    sell_str = 0.5 * mfi_sell + 0.3 * bearish_div + 0.2 * (-ind["trend_score"]).clip(0, None)

    raw = (buy_str - sell_str) * 100
    # Oversold correction
    oversold = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    raw = raw + oversold * above_200 * 35 - oversold * (1 - above_200) * 10
    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def main():
    print("=" * 110)
    print("  CSI BATCH 2: v23-v32 — Hybrid Combinations of Batch 1 Winners")
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
        ("v8+1 BASELINE", v8_baseline),
        ("v17 MFI (Batch1 champ)", v17_mfi_divergence),
        ("v23 MFI+Structure", v23_mfi_structure),
        ("v24 MFI+v8Corrections", v24_mfi_v8_corrections),
        ("v25 MR+MFI", v25_mr_mfi),
        ("v26 Structure+Breakout", v26_structure_breakout),
        ("v27 Top4 Ensemble", v27_top4_ensemble),
        ("v28 Weighted Ensemble", v28_weighted_ensemble),
        ("v29 MFI Enhanced", v29_mfi_enhanced),
        ("v30 Regime Switch", v30_regime_switch),
        ("v31 Structure+v8Corr", v31_structure_v8corr),
        ("v32 Sell Specialist", v32_sell_specialist),
    ]

    all_results = []
    for name, fn in strategies:
        t = time.time()
        s = run_strategy(name, fn)
        s["time"] = round(time.time() - t, 1)
        all_results.append(s)

    print(f"\n\n  {'='*80}")
    print(f"  BATCH 2 LEADERBOARD")
    print(f"  {'='*80}")
    ranked = sorted(all_results, key=lambda x: x.get("combined", 0) if not np.isnan(x.get("combined", 0)) else -999, reverse=True)
    for i, s in enumerate(ranked):
        tag = ""
        if s["name"] not in ("v8+1 BASELINE", "v17 MFI (Batch1 champ)"):
            if i == 0:
                tag = " ** NEW CHAMPION **"
        print(f"  #{i+1:2d}: {s['name']:<28} C={s['combined']:6.3f}  Sh={s['med_sh']:6.3f}  Spr={s['spread']:+5.1f}%  "
              f"GS={s.get('good_sep','?')}{tag}")

    # Save
    try:
        with open(LEADERBOARD_FILE, "r") as f:
            existing = json.load(f)
    except Exception:
        existing = []
    for s in all_results:
        if s["name"] not in ("v8+1 BASELINE", "v17 MFI (Batch1 champ)"):
            entry = {k: v for k, v in s.items() if not isinstance(v, float) or not np.isnan(v)}
            existing.append(entry)
    existing.sort(key=lambda x: x.get("combined", 0), reverse=True)
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"\n  Leaderboard updated ({len(existing)} entries)")
    print(f"  Total time: {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
