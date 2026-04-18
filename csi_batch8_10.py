#!/usr/bin/env python3
"""
CSI BATCH 8-10: v73-v100 — Final Tournament
=============================================
TOP 3 CHAMPIONS:
  v64 Consec NoSmooth   C=0.669 Sh=0.607 Spr=+10.2% GS=55/98 (NO EMA!)
  v67 Consec StrongMR   C=0.667 Sh=0.621 Spr=+7.4%  GS=60/98 (stronger MR)
  v66 Consec LightFlow  C=0.645 Sh=0.602 Spr=+7.2%  GS=63/98 (less vol flow)

KEY INSIGHT: EMA smoothing HURTS consecutive pattern signals!
  - v49 raw signal already has structure (discrete events)
  - Smoothing blurs the crisp buy/sell boundaries
  - Stronger mean-reversion weights exploit the core pattern better
"""

import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi_mega_harness import (
    UNIVERSE, preload_all, get_indicators, evaluate, summarize, run_strategy,
    v17_mfi_divergence, LEADERBOARD_FILE,
)


def _count_runs(series):
    vals = series.values.astype(float)
    out = np.zeros(len(vals))
    for i in range(1, len(vals)):
        if vals[i] > 0:
            out[i] = out[i-1] + 1
    return pd.Series(out, index=series.index)


# ═══════════════════════════════════════════════════════════
# BATCH 8: v73-v82 — Combine v64/v67/v66 insights
# ═══════════════════════════════════════════════════════════

def v64_ref(ind):
    """v64 reference: NoSmooth (C=0.669 champion)."""
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


def v73_nosmooth_strongmr(ind):
    """v73: Combine v64 (no smooth) + v67 (strong MR 0.7) — best of both."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mr_sig = dn_exhaust * 0.7 - up_exhaust * 0.7  # Strong MR from v67
    mom_sig = up_mom * 0.2 - dn_mom * 0.2
    raw = (mr_sig + mom_sig + 0.30 * ind["vol_flow"] + 0.20 * ind["trend_score"]) * 80
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)  # No smooth from v64


def v74_nosmooth_lightflow(ind):
    """v74: No smooth + light vol flow (15%) from v66."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mr_sig = dn_exhaust * 0.6 - up_exhaust * 0.6  # v66 strength
    mom_sig = up_mom * 0.35 - dn_mom * 0.35
    raw = (mr_sig + mom_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v75_triple_fusion(ind):
    """v75: All 3 insights: no smooth + strong MR 0.7 + light flow 15%."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mr_sig = dn_exhaust * 0.7 - up_exhaust * 0.7  # Strong MR
    mom_sig = up_mom * 0.2 - dn_mom * 0.2
    raw = (mr_sig + mom_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85  # Light flow
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)  # No smooth


def v76_triple_no_trend(ind):
    """v76: Triple fusion but NO trend component (pure pattern + minimal flow)."""
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
    raw = (mr_sig + mom_sig + 0.10 * ind["vol_flow"]) * 90
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v77_pure_pattern(ind):
    """v77: Absolutely pure consecutive pattern — zero indicator overlay."""
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
    mom_sig = up_mom * 0.25 - dn_mom * 0.25
    raw = (mr_sig + mom_sig) * 100
    raw = raw + 1
    return raw.clip(-100, 100)


def v78_pure_pattern_bias3(ind):
    """v78: Pure pattern + bias 3."""
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
    mom_sig = up_mom * 0.25 - dn_mom * 0.25
    raw = (mr_sig + mom_sig) * 100 + 3
    return raw.clip(-100, 100)


def v79_consec_mr_sweep(ind):
    """v79: Sweep MR=0.8 (maximum mean-reversion emphasis)."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mr_sig = dn_exhaust * 0.8 - up_exhaust * 0.8
    mom_sig = up_mom * 0.15 - dn_mom * 0.15
    raw = (mr_sig + mom_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v80_consec_5plus_exhaust(ind):
    """v80: Exhaustion only at 5+ days (more selective, higher conviction)."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 5).astype(float) * ((up_runs - 4) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 5).astype(float) * ((dn_runs - 4) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 4)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 4)).astype(float)
    mr_sig = dn_exhaust * 0.7 - up_exhaust * 0.7
    mom_sig = up_mom * 0.25 - dn_mom * 0.25
    raw = (mr_sig + mom_sig + 0.20 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v81_consec_3plus_exhaust(ind):
    """v81: Exhaustion at 3+ with strong MR and no smooth."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 3).astype(float) * ((up_runs - 2) / 4).clip(0, 1)
    dn_exhaust = (dn_runs >= 3).astype(float) * ((dn_runs - 2) / 4).clip(0, 1)
    up_mom = (up_runs == 2).astype(float)
    dn_mom = (dn_runs == 2).astype(float)
    mr_sig = dn_exhaust * 0.7 - up_exhaust * 0.7
    mom_sig = up_mom * 0.2 - dn_mom * 0.2
    raw = (mr_sig + mom_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v82_asymmetric_exhaust(ind):
    """v82: Asymmetric exhaustion: sell exhaustion (up) more aggressive than buy (dn)."""
    close = ind["close"]
    above_200 = ind["above_200"].fillna(0)
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    # Up exhaustion: detect distribution tops (less aggressive sell)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1) * 0.5
    # Down exhaustion: detect capitulation bottoms (more aggressive buy in uptrend)
    dn_exhaust_buy = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1) * 0.8 * above_200
    dn_exhaust_sell = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1) * 0.3 * (1 - above_200)

    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mom_sig = up_mom * 0.25 - dn_mom * 0.25

    raw = (dn_exhaust_buy - up_exhaust - dn_exhaust_sell + mom_sig +
           0.20 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85
    raw = raw + above_200 * 5 + 1
    return raw.clip(-100, 100)


# ═══════════════════════════════════════════════════════════
# BATCH 9: v83-v92 — Micro-optimization of best variants
# ═══════════════════════════════════════════════════════════

def v83_nosmooth_strongmr_light(ind):
    """v83: No smooth + strong MR 0.7 + light flow 10% + light trend 10%."""
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
    raw = (mr_sig + mom_sig + 0.10 * ind["vol_flow"] + 0.10 * ind["trend_score"]) * 90
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v84_nosmooth_mr08(ind):
    """v84: No smooth + MR=0.8 + light overlays."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mr_sig = dn_exhaust * 0.8 - up_exhaust * 0.8
    mom_sig = up_mom * 0.15 - dn_mom * 0.15
    raw = (mr_sig + mom_sig + 0.10 * ind["vol_flow"] + 0.10 * ind["trend_score"]) * 90
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v85_pure_mr07_noflow(ind):
    """v85: Pure pattern MR=0.7 + NO flow, NO trend (maximally pure)."""
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
    raw = (mr_sig + mom_sig) * 100
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v86_scale_sweep_70(ind):
    """v86: Scaling factor 70 (vs default 80-100)."""
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
    raw = (mr_sig + mom_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 70
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v87_scale_sweep_100(ind):
    """v87: Scaling factor 100."""
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
    raw = (mr_sig + mom_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 100
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v88_bias_sweep_0(ind):
    """v88: Zero bias (no above_200 bonus, no constant)."""
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
    raw = (mr_sig + mom_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85
    return raw.clip(-100, 100)


def v89_bias_sweep_10(ind):
    """v89: Aggressive bias (+10 above_200 + 2 constant)."""
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
    raw = (mr_sig + mom_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85
    raw = raw + ind["above_200"].fillna(0) * 10 + 2
    return raw.clip(-100, 100)


def v90_mom_range_1_4(ind):
    """v90: Momentum continuation on runs 1-4 (include day 1 as continuation signal)."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 5).astype(float) * ((up_runs - 4) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 5).astype(float) * ((dn_runs - 4) / 3).clip(0, 1)
    up_mom = ((up_runs >= 1) & (up_runs <= 4)).astype(float)
    dn_mom = ((dn_runs >= 1) & (dn_runs <= 4)).astype(float)
    mr_sig = dn_exhaust * 0.7 - up_exhaust * 0.7
    mom_sig = up_mom * 0.2 - dn_mom * 0.2
    raw = (mr_sig + mom_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v91_mr_only(ind):
    """v91: Mean reversion ONLY — no momentum continuation at all."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    mr_sig = dn_exhaust * 0.8 - up_exhaust * 0.8
    raw = (mr_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v92_mom_only(ind):
    """v92: Momentum continuation ONLY — no mean-reversion at all."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mom_sig = up_mom * 0.5 - dn_mom * 0.5
    raw = (mom_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


# ═══════════════════════════════════════════════════════════
# BATCH 10: v93-v100 — Final Champion Tournament
# ═══════════════════════════════════════════════════════════

def v93_consec_mfi_sell_nosmooth(ind):
    """v93: NoSmooth + StrongMR + MFI sell overlay."""
    close = ind["close"]; mfi = ind["mfi"]
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
    raw = (mr_sig + mom_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85

    # MFI sell overlay
    price_high = (close >= close.rolling(20).max() * 0.98).astype(float)
    mfi_weak = (mfi < mfi.rolling(20).max() * 0.90).astype(float)
    raw = raw - price_high * mfi_weak * 10

    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v94_consec_oversold_boost(ind):
    """v94: NoSmooth + StrongMR + oversold-in-uptrend correction."""
    close = ind["close"]; rsi = ind["rsi"]
    above_200 = ind["above_200"].fillna(0)
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
    raw = (mr_sig + mom_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85

    # Oversold in uptrend boost
    oversold = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    raw = raw + oversold * above_200 * 20

    raw = raw + above_200 * 5 + 1
    return raw.clip(-100, 100)


def v95_gap_pattern(ind):
    """v95: Consecutive patterns with gap detection (large moves count double)."""
    close = ind["close"]; vol_20 = ind["vol_20"]
    ret = ind["ret_1"]

    # Detect gaps: moves > 1.5 sigma
    is_gap = (ret.abs() > 1.5 * vol_20).astype(float)

    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    # Weight by gap presence
    up_weighted = up * (1 + is_gap * 0.5)
    dn_weighted = dn * (1 + is_gap * 0.5)

    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1) * (1 + is_gap * 0.3)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1) * (1 + is_gap * 0.3)

    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mr_sig = dn_exhaust * 0.7 - up_exhaust * 0.7
    mom_sig = up_mom * 0.2 - dn_mom * 0.2
    raw = (mr_sig + mom_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v96_consec_vol_dampened(ind):
    """v96: NoSmooth + StrongMR + vol dampener (reduce in high vol)."""
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
    raw = (mr_sig + mom_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85
    raw = raw * ind["vol_dampener"]
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v97_consec_adaptive_scale(ind):
    """v97: Adaptive scaling based on recent pattern hit rate."""
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

    # Adaptive: use efficiency ratio to scale
    eff = ind["efficiency"].fillna(0.5)
    # In efficient/trending markets: patterns less effective (random walk breaks pattern)
    # In inefficient/mean-reverting markets: patterns more effective
    scale = 1.2 - 0.4 * eff  # 0.8 to 1.2

    raw = (mr_sig + mom_sig + 0.15 * ind["vol_flow"] + 0.15 * ind["trend_score"]) * 85 * scale
    raw = raw + ind["above_200"].fillna(0) * 5 + 1
    return raw.clip(-100, 100)


def v98_ensemble_top3_nosmooth(ind):
    """v98: Average of v64 + v73 + v75 (top NoSmooth variants)."""
    s1 = v64_ref(ind)
    s2 = v73_nosmooth_strongmr(ind)
    s3 = v75_triple_fusion(ind)
    return ((s1 + s2 + s3) / 3).clip(-100, 100)


def v99_best_features(ind):
    """v99: NoSmooth + StrongMR 0.75 + flow 12% + trend 12% + above200*6 + bias 1."""
    close = ind["close"]
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)
    mr_sig = dn_exhaust * 0.75 - up_exhaust * 0.75
    mom_sig = up_mom * 0.18 - dn_mom * 0.18
    raw = (mr_sig + mom_sig + 0.12 * ind["vol_flow"] + 0.12 * ind["trend_score"]) * 90
    raw = raw + ind["above_200"].fillna(0) * 6 + 1
    return raw.clip(-100, 100)


def v100_ultimate(ind):
    """v100: The ultimate CSI — combination of every proven insight.
    - Consecutive patterns (v49 core insight)
    - No EMA smoothing (v64 insight)
    - Strong mean-reversion 0.7 (v67 insight)
    - Light overlays (v66 insight)
    - MFI sell overlay (v17 insight)
    - Oversold-in-uptrend correction (v8 insight)
    """
    close = ind["close"]; rsi = ind["rsi"]; mfi = ind["mfi"]
    above_200 = ind["above_200"].fillna(0)

    # Core: consecutive patterns
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    up_runs = _count_runs(up)
    dn_runs = _count_runs(dn)
    up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
    dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)
    up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
    dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)

    # Strong MR + moderate momentum
    mr_sig = dn_exhaust * 0.7 - up_exhaust * 0.7
    mom_sig = up_mom * 0.2 - dn_mom * 0.2

    # Light overlays
    raw = (mr_sig + mom_sig + 0.12 * ind["vol_flow"] + 0.12 * ind["trend_score"]) * 88

    # MFI sell overlay (selective)
    price_high = (close >= close.rolling(20).max() * 0.98).astype(float)
    mfi_weak = (mfi < mfi.rolling(20).max() * 0.90).astype(float)
    raw = raw - price_high * mfi_weak * 8

    # Oversold in uptrend correction
    oversold = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    raw = raw + oversold * above_200 * 15

    raw = raw + above_200 * 5 + 1
    return raw.clip(-100, 100)  # NO smoothing


def main():
    print("=" * 110)
    print("  CSI BATCH 8-10: v73-v100 — FINAL TOURNAMENT")
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
        ("v64 NoSmooth (champion)", v64_ref),
        # Batch 8
        ("v73 NoSmooth+StrongMR", v73_nosmooth_strongmr),
        ("v74 NoSmooth+LightFlow", v74_nosmooth_lightflow),
        ("v75 Triple Fusion", v75_triple_fusion),
        ("v76 Triple NoTrend", v76_triple_no_trend),
        ("v77 Pure Pattern", v77_pure_pattern),
        ("v78 Pure Pattern+Bias3", v78_pure_pattern_bias3),
        ("v79 MR=0.8 Sweep", v79_consec_mr_sweep),
        ("v80 5+Day Exhaust", v80_consec_5plus_exhaust),
        ("v81 3+Day Exhaust+NoSmth", v81_consec_3plus_exhaust),
        ("v82 Asymmetric Exhaust", v82_asymmetric_exhaust),
        # Batch 9
        ("v83 NS+MR07+Lt10", v83_nosmooth_strongmr_light),
        ("v84 NS+MR08+Lt10", v84_nosmooth_mr08),
        ("v85 Pure MR07 NoFlow", v85_pure_mr07_noflow),
        ("v86 Scale=70", v86_scale_sweep_70),
        ("v87 Scale=100", v87_scale_sweep_100),
        ("v88 Bias=0", v88_bias_sweep_0),
        ("v89 Bias=10+2", v89_bias_sweep_10),
        ("v90 Mom 1-4 Range", v90_mom_range_1_4),
        ("v91 MR Only", v91_mr_only),
        ("v92 Mom Only", v92_mom_only),
        # Batch 10
        ("v93 NS+MR+MFIsell", v93_consec_mfi_sell_nosmooth),
        ("v94 NS+MR+OversoldBoost", v94_consec_oversold_boost),
        ("v95 Gap Pattern", v95_gap_pattern),
        ("v96 Vol Dampened", v96_consec_vol_dampened),
        ("v97 Adaptive Scale", v97_consec_adaptive_scale),
        ("v98 Ensemble Top3 NS", v98_ensemble_top3_nosmooth),
        ("v99 Best Features", v99_best_features),
        ("v100 ULTIMATE", v100_ultimate),
    ]

    all_results = []
    for name, fn in strategies:
        t = time.time()
        s = run_strategy(name, fn)
        s["time"] = round(time.time() - t, 1)
        all_results.append(s)

    print(f"\n\n  {'='*95}")
    print(f"  FINAL TOURNAMENT LEADERBOARD (v73-v100)")
    print(f"  {'='*95}")
    ranked = sorted(all_results, key=lambda x: x.get("combined", 0) if not np.isnan(x.get("combined", 0)) else -999, reverse=True)
    for i, s in enumerate(ranked):
        tag = ""
        if s["name"] != "v64 NoSmooth (champion)" and i == 0:
            tag = " ** NEW OVERALL CHAMPION **"
        print(f"  #{i+1:2d}: {s['name']:<28} C={s['combined']:6.3f}  Sh={s['med_sh']:6.3f}  Spr={s['spread']:+5.1f}%  "
              f"GS={s.get('good_sep','?')}{tag}")

    # Grand leaderboard
    try:
        with open(LEADERBOARD_FILE, "r") as f:
            existing = json.load(f)
    except Exception:
        existing = []
    existing_names = {e["name"] for e in existing}
    for s in all_results:
        if s["name"] != "v64 NoSmooth (champion)" and s["name"] not in existing_names:
            entry = {k: v for k, v in s.items() if not isinstance(v, float) or not np.isnan(v)}
            existing.append(entry)
    existing.sort(key=lambda x: x.get("combined", 0), reverse=True)
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    print(f"\n\n  {'='*95}")
    print(f"  GRAND ALL-TIME TOP 10 (all 100 versions)")
    print(f"  {'='*95}")
    for i, s in enumerate(existing[:10]):
        print(f"  #{i+1:2d}: {s['name']:<28} C={s.get('combined',0):6.3f}  Sh={s.get('med_sh',0):6.3f}  "
              f"Spr={s.get('spread',0):+5.1f}%  GS={s.get('good_sep','?')}")

    print(f"\n  Leaderboard: {len(existing)} entries total")
    print(f"  Total time: {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
