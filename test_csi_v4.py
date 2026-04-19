"""
CSI v4 Test Harness — Exhaustion/Extension Sell Architecture
=============================================================
v3 PROBLEM: sell signals fire when momentum is negative (at bottoms → mean reversion bounce).
v4 FIX: sell signals fire when price is EXTENDED above MA and momentum is EXHAUSTING (at tops).

Key insight: In a long-biased market:
  - BUY at oversold + bounce starting (dip buying) → works (v3 buy hit 56.7%)
  - SELL at "everything looks bad" → WRONG (catches bounces, v3 sell avg +0.87%)
  - SELL at "euphoria + exhaustion" → should work (mean reversion from ABOVE)

Sell side redesign:
  1. Extension above MA (how far above SMA20/50, percentile rank)
  2. Momentum exhaustion (positive momentum that's decelerating)
  3. RSI/Stoch overbought + declining (kept from v3)
  4. Volume-price divergence (price up, volume declining)
  5. MACD histogram declining from positive peak
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd

TEST_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "CRM", "ADBE", "NFLX", "CRWD",
    "JPM", "BAC", "GS", "MS", "SCHW", "AFRM",
    "JNJ", "UNH", "PFE", "ABBV", "MRNA",
    "LMT", "RTX", "NOC", "GD",
    "CAT", "DE", "BA", "UPS", "GE",
    "XOM", "CVX", "COP", "SLB",
    "HD", "NKE", "SBUX", "PG", "KO", "COST",
    "UPST", "IONQ", "DKNG", "SNAP",
    "SPY", "QQQ", "IWM", "DIA",
    "GLD", "SLV", "BTC-USD",
]

PRICES_DIR = os.path.join(SRC_DIR, "data", "prices")


def load_ohlcv(symbol):
    for suffix in ["_1d.csv", ".csv"]:
        path = os.path.join(PRICES_DIR, f"{symbol}{suffix}")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]
            for dc in ["date", "datetime", "timestamp"]:
                if dc in df.columns:
                    df["date"] = pd.to_datetime(df[dc]).dt.strftime("%Y-%m-%d")
                    break
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["close", "high", "low", "volume"])
            return df
    return pd.DataFrame()


def compute_csi_v4(df):
    """CSI v4: Same buy side as v3, completely new sell side based on exhaustion/extension."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    n = len(close)

    if n < 60:
        return pd.Series(dtype=float)

    # ── 1. TREND CONTEXT (same as v3) ──────────────────────
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()

    above_20 = (close > sma_20).astype(float)
    above_50 = (close > sma_50).astype(float)

    ma20_slope = sma_20.pct_change(5)
    ma20_slope_n = (ma20_slope / ma20_slope.abs().rolling(60).max().replace(0, float("nan"))).clip(-1, 1)
    ma50_slope = sma_50.pct_change(10)
    ma50_slope_n = (ma50_slope / ma50_slope.abs().rolling(60).max().replace(0, float("nan"))).clip(-1, 1)

    ma_context = (0.25 * (above_20 * 2 - 1) + 0.25 * (above_50 * 2 - 1) +
                  0.25 * ma20_slope_n + 0.25 * ma50_slope_n)

    # ── 2. MULTI-TF MOMENTUM (same as v3) ─────────────────
    ret_1 = close.pct_change(1)
    ret_5 = close.pct_change(5)
    ret_10 = close.pct_change(10)
    ret_20 = close.pct_change(20)

    vol_20 = ret_1.rolling(20).std().replace(0, float("nan"))
    mom_fast = (ret_5 / (vol_20 * np.sqrt(5))).clip(-3, 3) / 3
    mom_med = (ret_10 / (vol_20 * np.sqrt(10))).clip(-3, 3) / 3
    mom_slow = (ret_20 / (vol_20 * np.sqrt(20))).clip(-3, 3) / 3

    raw_mom = 0.50 * mom_fast + 0.30 * mom_med + 0.20 * mom_slow
    signs_mom = pd.concat([np.sign(mom_fast), np.sign(mom_med), np.sign(mom_slow)], axis=1)
    agreement = signs_mom.sum(axis=1).abs() / 3.0
    mom_score = raw_mom * (0.5 + 0.5 * agreement)

    mom_accel = mom_fast - mom_fast.shift(5)
    mom_accel_n = (mom_accel / mom_accel.abs().rolling(30).max().replace(0, float("nan"))).clip(-1, 1)

    # ── 3. VOLUME FLOW (same as v3) ───────────────────────
    vol_sma20 = volume.rolling(20).mean().replace(0, float("nan"))

    up_vol = volume.where(close > close.shift(1), 0.0).rolling(10).sum()
    dn_vol = volume.where(close <= close.shift(1), 0.0).rolling(10).sum()
    vol_ratio = (up_vol - dn_vol) / (up_vol + dn_vol).replace(0, float("nan"))
    vol_ratio = vol_ratio.clip(-1, 1)

    obv_dir = np.sign(close.diff()).fillna(0)
    obv_raw = (obv_dir * volume).cumsum()
    obv_ema_f = obv_raw.ewm(span=10, adjust=False).mean()
    obv_ema_s = obv_raw.ewm(span=30, adjust=False).mean()
    obv_diff = obv_ema_f - obv_ema_s
    obv_range = obv_diff.abs().rolling(40).max().replace(0, float("nan"))
    obv_signal = (obv_diff / obv_range).clip(-1, 1)

    clv = ((close - low) - (high - close)) / (high - low).replace(0, float("nan"))
    ad_line = (clv * volume).cumsum()
    ad_ema5 = ad_line.ewm(span=5, adjust=False).mean()
    ad_ema20 = ad_line.ewm(span=20, adjust=False).mean()
    ad_osc = ad_ema5 - ad_ema20
    ad_rng = ad_osc.abs().rolling(40).max().replace(0, float("nan"))
    ad_score = (ad_osc / ad_rng).clip(-1, 1)

    vol_flow = 0.40 * vol_ratio + 0.30 * obv_signal + 0.30 * ad_score

    # ── 4. OSCILLATOR BUY: Oversold bounce (same as v3) ───
    delta_c = close.diff()
    gain_c = delta_c.clip(lower=0).rolling(14).mean()
    loss_c = (-delta_c.clip(upper=0)).rolling(14).mean()
    rs_c = gain_c / loss_c.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs_c))

    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14).replace(0, float("nan"))
    stoch_k = stoch_k.rolling(3).mean()

    rsi_buy = np.where(rsi < 35, (35 - rsi) / 35, 0.0)
    stoch_buy = np.where(stoch_k < 25, (25 - stoch_k) / 25, 0.0)
    osc_buy = pd.Series(0.55 * rsi_buy + 0.45 * stoch_buy, index=close.index).clip(0, 1)

    # ── 4b. OSCILLATOR SELL: Overbought + declining (same) ─
    rsi_declining = (rsi < rsi.shift(3)).astype(float)
    stoch_declining = (stoch_k < stoch_k.shift(3)).astype(float)
    rsi_sell = np.where((rsi > 70) & (rsi_declining > 0), (rsi - 70) / 30, 0.0)
    stoch_sell = np.where((stoch_k > 80) & (stoch_declining > 0), (stoch_k - 80) / 20, 0.0)
    osc_sell = pd.Series(0.55 * rsi_sell + 0.45 * stoch_sell, index=close.index).clip(0, 1)

    # ── 5. MACD + ADX (same as v3) ────────────────────────
    ema12_c = close.ewm(span=12, adjust=False).mean()
    ema26_c = close.ewm(span=26, adjust=False).mean()
    macd_line_c = ema12_c - ema26_c
    signal_c = macd_line_c.ewm(span=9, adjust=False).mean()
    hist_c = macd_line_c - signal_c
    hist_range = hist_c.abs().rolling(20).max().replace(0, float("nan"))
    macd_n = (hist_c / hist_range).clip(-1, 1)
    hist_accel = hist_c.diff(3)
    macd_accel = (hist_accel / hist_range).clip(-1, 1)
    trend_macd = 0.6 * macd_n + 0.4 * macd_accel

    tr_c = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    atr14 = tr_c.ewm(span=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    adx_val = dx.ewm(span=14, adjust=False).mean()
    di_diff = (plus_di - minus_di) / (plus_di + minus_di).replace(0, float("nan"))
    adx_regime = ((adx_val - 15) / 35).clip(0, 1)
    trend_adx = di_diff * adx_regime
    trend_score = 0.55 * trend_macd + 0.45 * trend_adx

    # ── 6. VOLATILITY CONTEXT (same as v3) ─────────────────
    vol_pct = vol_20.rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
    vol_dampener = 1.0 - 0.3 * (vol_pct - 0.5).clip(0, 0.5)

    # ════════════════════════════════════════════════════════
    # ═══ NEW SELL SIDE: EXHAUSTION + EXTENSION DETECTOR ════
    # ════════════════════════════════════════════════════════
    # Fire when price is EXTENDED above MA and internals are WEAKENING.
    # NOT when price is below MA (that's oversold territory → bounce).

    # S1. Extension above moving averages (mean reversion pressure from above)
    # How far above SMA20/50 is the price? Higher = more reversion pressure.
    ext_20 = (close / sma_20 - 1).clip(0, None)  # pct above SMA20 (0 when below)
    ext_50 = (close / sma_50 - 1).clip(0, None)  # pct above SMA50 (0 when below)
    # Adaptive normalization: percentile rank within history
    ext_20_rank = ext_20.rolling(252, min_periods=60).rank(pct=True).fillna(0)
    ext_50_rank = ext_50.rolling(252, min_periods=60).rank(pct=True).fillna(0)
    # Combined extension: 0 = at/below MA, 1 = at historical max extension
    extension_raw = 0.50 * ext_20_rank + 0.50 * ext_50_rank
    # Only activate when extension is notable (>50th percentile = above median extension)
    extension_score = ((extension_raw - 0.5) * 2).clip(0, 1)

    # S2. Momentum exhaustion: positive momentum that's decelerating
    # Fires when momentum WAS positive and is now declining (the turn from bullish)
    mom_avg_5 = mom_score.rolling(5).mean()
    mom_was_positive = (mom_avg_5.shift(5) > 0.05).astype(float)
    mom_now_lower = (mom_score < mom_avg_5.shift(5)).astype(float)
    mom_drop_size = (mom_avg_5.shift(5) - mom_score).clip(0, None)
    mom_exhaust = (mom_drop_size / 0.3).clip(0, 1) * mom_was_positive * mom_now_lower

    # S3. RSI/Stoch overbought + declining (from v3 osc_sell, correct design)
    # Already computed above as osc_sell

    # S4. Volume-price divergence: price advancing but volume declining
    # Strong sell signal: market rallying on fading volume = weak conviction
    price_up_10d = (close > close.shift(10)).astype(float)
    vol_10ma = volume.rolling(10).mean()
    vol_20ma = volume.rolling(20).mean()
    vol_weakening = (vol_10ma < vol_20ma * 0.92).astype(float)  # vol declining >8%
    vol_price_div = (price_up_10d * vol_weakening).rolling(5).mean()

    # S5. MACD histogram declining from positive peak
    # Momentum peaked and is now fading even though it may still be positive
    hist_pos = (hist_c > 0).astype(float)
    hist_peak_10 = hist_c.rolling(10).max()
    hist_off_peak = ((hist_peak_10 - hist_c) / hist_peak_10.abs().replace(0, float("nan"))).clip(0, 1)
    # Only count when histogram was meaningfully positive
    macd_exhaust = hist_off_peak * hist_pos * (hist_peak_10 > 0).astype(float)

    # ── MASTER BLEND ───────────────────────────────────────
    # BUY SCORE (same as v3: trend + momentum + dip buy + accumulation)
    buy_raw = (
        0.30 * trend_score.clip(0, None) +
        0.25 * mom_score.clip(0, None) +
        0.15 * osc_buy +
        0.15 * vol_flow.clip(0, None) +
        0.10 * ma_context.clip(0, None) +
        0.05 * mom_accel_n.clip(0, None)
    )

    # SELL SCORE (NEW: exhaustion + extension, NOT lagging weakness)
    sell_raw = (
        0.30 * extension_score +       # price extended above MA
        0.25 * mom_exhaust +           # positive momentum fading
        0.20 * osc_sell +              # overbought + declining oscillators
        0.15 * vol_price_div +         # volume-price divergence
        0.10 * macd_exhaust            # MACD histogram off peak
    )

    raw_csi = (buy_raw - sell_raw) * 100

    # Volume confirmation
    vol_confirm = (np.sign(raw_csi) * np.sign(vol_flow)).clip(0, 1)
    raw_csi = raw_csi * (0.80 + 0.20 * vol_confirm)
    raw_csi = raw_csi * vol_dampener

    # Signal gating
    buy_factors = pd.concat([
        (trend_score > 0.05).astype(float),
        (mom_score > 0.05).astype(float),
        (vol_flow > 0.05).astype(float),
        (ma_context > 0).astype(float),
    ], axis=1).sum(axis=1)
    sell_ex_factors = pd.concat([
        (extension_score > 0.15).astype(float),
        (mom_exhaust > 0.10).astype(float),
        (osc_sell > 0.10).astype(float),
        (vol_price_div > 0.15).astype(float),
        (macd_exhaust > 0.15).astype(float),
    ], axis=1).sum(axis=1)

    buy_gate = (buy_factors >= 2).astype(float)
    sell_gate = (sell_ex_factors >= 2).astype(float)
    gate = np.where(raw_csi > 0, buy_gate, np.where(raw_csi < 0, sell_gate, 0.5))
    gate = pd.Series(gate, index=close.index)
    raw_csi = raw_csi * (0.3 + 0.7 * gate)
    raw_csi = raw_csi.clip(-100, 100)

    # Adaptive smoothing
    ema_f = raw_csi.ewm(span=3, adjust=False).mean()
    ema_s = raw_csi.ewm(span=8, adjust=False).mean()
    sw = adx_regime.fillna(0.5)
    csi = sw * ema_f + (1 - sw) * ema_s

    return csi


def evaluate_csi(symbol, csi, close):
    valid = csi.notna() & close.notna()
    csi = csi[valid]
    close = close[valid]
    if len(csi) < 60:
        return {"symbol": symbol, "error": "insufficient data"}

    fwd_5 = close.pct_change(5).shift(-5)
    fwd_10 = close.pct_change(10).shift(-10)
    fwd_1 = close.pct_change(1).shift(-1)

    mask = fwd_10.notna() & csi.notna()
    csi_v = csi[mask]
    fwd_5_v = fwd_5[mask]
    fwd_10_v = fwd_10[mask]
    fwd_1_v = fwd_1[mask]
    if len(csi_v) < 30:
        return {"symbol": symbol, "error": "insufficient valid data"}

    corr_5d = csi_v.corr(fwd_5_v)
    corr_10d = csi_v.corr(fwd_10_v)
    dir_correct_5d = ((np.sign(csi_v) == np.sign(fwd_5_v)) | (csi_v == 0)).mean()
    dir_correct_10d = ((np.sign(csi_v) == np.sign(fwd_10_v)) | (csi_v == 0)).mean()

    buy_mask = csi_v > 30
    sell_mask = csi_v < -30
    strong_buy = csi_v > 50
    strong_sell = csi_v < -50
    buy_count = buy_mask.sum()
    sell_count = sell_mask.sum()

    buy_avg_5d = fwd_5_v[buy_mask].mean() * 100 if buy_count > 5 else np.nan
    sell_avg_5d = fwd_5_v[sell_mask].mean() * 100 if sell_count > 5 else np.nan
    sb_avg = fwd_5_v[strong_buy].mean() * 100 if strong_buy.sum() > 3 else np.nan
    ss_avg = fwd_5_v[strong_sell].mean() * 100 if strong_sell.sum() > 3 else np.nan

    buy_hit = (fwd_5_v[buy_mask] > 0).mean() if buy_count > 5 else np.nan
    sell_hit = (fwd_5_v[sell_mask] < 0).mean() if sell_count > 5 else np.nan

    position = np.sign(csi_v).clip(0, 1)
    strat_ret = (position * fwd_1_v).dropna()
    buy_hold_ret = fwd_1_v.dropna()

    sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) if len(strat_ret) > 20 and strat_ret.std() > 0 else np.nan
    bh_sharpe = (buy_hold_ret.mean() / buy_hold_ret.std()) * np.sqrt(252) if buy_hold_ret.std() > 0 else np.nan

    cumret = (1 + strat_ret).cumprod()
    peak = cumret.cummax()
    dd = (cumret - peak) / peak
    max_dd = dd.min() * 100
    total_ret = cumret.iloc[-1] / cumret.iloc[0] if len(cumret) > 0 and cumret.iloc[0] > 0 else 1
    years = len(strat_ret) / 252
    cagr = (total_ret ** (1 / max(years, 0.1)) - 1) * 100

    signal_pct = ((csi_v.abs() > 30).sum() / len(csi_v)) * 100

    return {
        "symbol": symbol, "n_bars": len(csi_v),
        "corr_5d": round(corr_5d, 4) if not np.isnan(corr_5d) else None,
        "corr_10d": round(corr_10d, 4) if not np.isnan(corr_10d) else None,
        "dir_acc_5d": round(dir_correct_5d * 100, 1),
        "dir_acc_10d": round(dir_correct_10d * 100, 1),
        "buy_count": int(buy_count), "sell_count": int(sell_count),
        "buy_avg_5d_pct": round(buy_avg_5d, 3) if not np.isnan(buy_avg_5d) else None,
        "sell_avg_5d_pct": round(sell_avg_5d, 3) if not np.isnan(sell_avg_5d) else None,
        "strong_buy_avg": round(sb_avg, 3) if not np.isnan(sb_avg) else None,
        "strong_sell_avg": round(ss_avg, 3) if not np.isnan(ss_avg) else None,
        "buy_hit_rate": round(buy_hit * 100, 1) if buy_hit is not None and not np.isnan(buy_hit) else None,
        "sell_hit_rate": round(sell_hit * 100, 1) if sell_hit is not None and not np.isnan(sell_hit) else None,
        "sharpe": round(sharpe, 3) if not np.isnan(sharpe) else None,
        "bh_sharpe": round(bh_sharpe, 3) if not np.isnan(bh_sharpe) else None,
        "max_dd_pct": round(max_dd, 1),
        "cagr_pct": round(cagr, 1) if not np.isnan(cagr) else None,
        "signal_freq_pct": round(signal_pct, 1),
    }


def main():
    print("=" * 100)
    print("  CSI v4 — EXHAUSTION/EXTENSION SELL ARCHITECTURE")
    print("  Sell signal: detect tops (extended + fading), NOT bottoms (already fallen)")
    print("=" * 100)
    print()

    results = []
    skipped = []

    for sym in TEST_UNIVERSE:
        df = load_ohlcv(sym)
        if df.empty or len(df) < 100:
            skipped.append(sym)
            continue

        df_r = df.reset_index(drop=True)
        for col in ["close", "high", "low", "volume"]:
            df_r[col] = df[col].astype(float).reset_index(drop=True)

        csi = compute_csi_v4(df_r)
        if csi.empty:
            skipped.append(sym)
            continue

        metrics = evaluate_csi(sym, csi, df_r["close"])
        if "error" in metrics:
            skipped.append(sym)
            continue
        results.append(metrics)

    if skipped:
        print(f"  Skipped ({len(skipped)}): {', '.join(skipped)}")
        print()

    print(f"{'Symbol':<10} {'Corr5d':>7} {'Corr10d':>8} {'DirAcc5':>8} {'BuyHit':>7} {'SellHit':>8} "
          f"{'BuyAvg5d':>9} {'SellAvg5d':>10} {'Sharpe':>7} {'B&H Sh':>7} {'SigFreq':>8}")
    print("-" * 110)

    for r in sorted(results, key=lambda x: x.get("sharpe") or -999, reverse=True):
        corr5 = f"{r['corr_5d']:.4f}" if r['corr_5d'] is not None else "   N/A"
        corr10 = f"{r['corr_10d']:.4f}" if r['corr_10d'] is not None else "   N/A"
        buy_hr = f"{r['buy_hit_rate']:.1f}%" if r['buy_hit_rate'] is not None else "  N/A"
        sell_hr = f"{r['sell_hit_rate']:.1f}%" if r['sell_hit_rate'] is not None else "  N/A"
        buy_avg = f"{r['buy_avg_5d_pct']:+.3f}%" if r['buy_avg_5d_pct'] is not None else "    N/A"
        sell_avg = f"{r['sell_avg_5d_pct']:+.3f}%" if r['sell_avg_5d_pct'] is not None else "     N/A"
        sharpe = f"{r['sharpe']:.3f}" if r['sharpe'] is not None else "  N/A"
        bh_sh = f"{r['bh_sharpe']:.3f}" if r['bh_sharpe'] is not None else "  N/A"
        print(f"{r['symbol']:<10} {corr5:>7} {corr10:>8} {r['dir_acc_5d']:>7.1f}% {buy_hr:>7} {sell_hr:>8} "
              f"{buy_avg:>9} {sell_avg:>10} {sharpe:>7} {bh_sh:>7} {r['signal_freq_pct']:>7.1f}%")

    print()
    print("=" * 100)
    print("  AGGREGATE STATISTICS")
    print("=" * 100)

    def safe_mean(vals):
        clean = [v for v in vals if v is not None and not np.isnan(v)]
        return np.mean(clean) if clean else float("nan")

    def safe_median(vals):
        clean = [v for v in vals if v is not None and not np.isnan(v)]
        return np.median(clean) if clean else float("nan")

    n = len(results)
    corr5_vals = [r["corr_5d"] for r in results]
    dir_acc_5d = [r["dir_acc_5d"] for r in results]
    buy_hits = [r["buy_hit_rate"] for r in results]
    sell_hits = [r["sell_hit_rate"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    bh_sharpes = [r["bh_sharpe"] for r in results]
    buy_avgs = [r["buy_avg_5d_pct"] for r in results]
    sell_avgs = [r["sell_avg_5d_pct"] for r in results]

    print(f"  Assets tested:           {n}")
    print(f"  Mean Corr (5d fwd):      {safe_mean(corr5_vals):.4f}")
    print(f"  Mean Dir Accuracy (5d):  {safe_mean(dir_acc_5d):.1f}%")
    print(f"  Mean Buy Hit Rate:       {safe_mean(buy_hits):.1f}%")
    print(f"  Mean Sell Hit Rate:      {safe_mean(sell_hits):.1f}%")
    print(f"  Median Sharpe:           {safe_median(sharpes):.3f}")
    print(f"  Mean Sharpe:             {safe_mean(sharpes):.3f}")
    print(f"  Median B&H Sharpe:       {safe_median(bh_sharpes):.3f}")
    print(f"  Avg Buy Signal 5d Ret:   {safe_mean(buy_avgs):.3f}%")
    print(f"  Avg Sell Signal 5d Ret:  {safe_mean(sell_avgs):.3f}%")

    sharpe_beat = sum(1 for r in results if r["sharpe"] is not None and r["bh_sharpe"] is not None and r["sharpe"] > r["bh_sharpe"])
    sharpe_eligible = sum(1 for r in results if r["sharpe"] is not None and r["bh_sharpe"] is not None)
    buy_profitable = sum(1 for r in results if r["buy_avg_5d_pct"] is not None and r["buy_avg_5d_pct"] > 0)
    buy_eligible = sum(1 for r in results if r["buy_avg_5d_pct"] is not None)
    sell_correct = sum(1 for r in results if r["sell_avg_5d_pct"] is not None and r["sell_avg_5d_pct"] < 0)
    sell_eligible = sum(1 for r in results if r["sell_avg_5d_pct"] is not None)

    print()
    print("  QUALITY BENCHMARKS:")
    print(f"    Sharpe beats B&H:       {sharpe_beat}/{sharpe_eligible} ({100*sharpe_beat/max(sharpe_eligible,1):.0f}%)")
    print(f"    Buy signals profitable: {buy_profitable}/{buy_eligible} ({100*buy_profitable/max(buy_eligible,1):.0f}%)")
    print(f"    Sell signals correct:   {sell_correct}/{sell_eligible} ({100*sell_correct/max(sell_eligible,1):.0f}%)")
    print(f"    Dir accuracy > 50%:     {sum(1 for d in dir_acc_5d if d > 50)}/{n}")

    sb_avgs = [r["strong_buy_avg"] for r in results if r["strong_buy_avg"] is not None]
    ss_avgs = [r["strong_sell_avg"] for r in results if r["strong_sell_avg"] is not None]
    print()
    print("  STRONG SIGNAL ANALYSIS (|CSI| > 50):")
    print(f"    Strong Buy avg 5d ret:  {np.mean(sb_avgs):.3f}% ({len(sb_avgs)} assets)" if sb_avgs else "    Strong Buy: insufficient data")
    print(f"    Strong Sell avg 5d ret: {np.mean(ss_avgs):.3f}% ({len(ss_avgs)} assets)" if ss_avgs else "    Strong Sell: insufficient data")

    print()
    print("  COMPARISON vs PREVIOUS VERSIONS:")
    print("  ┌────────────────────────┬──────────┬──────────┬──────────┐")
    print("  │ Metric                 │    v2    │    v3    │  v4 NOW  │")
    print("  ├────────────────────────┼──────────┼──────────┼──────────┤")
    print(f"  │ Mean Sell Hit Rate     │  38.9%   │  41.0%   │ {safe_mean(sell_hits):>5.1f}%   │")
    print(f"  │ Sell signals correct   │  13/50   │  11/50   │ {sell_correct:>2d}/{sell_eligible:<2d}    │")
    print(f"  │ Avg Sell 5d Ret        │ +0.890%  │ +0.867%  │{safe_mean(sell_avgs):>+7.3f}%  │")
    print(f"  │ Mean Buy Hit Rate      │  56.1%   │  56.7%   │ {safe_mean(buy_hits):>5.1f}%   │")
    print(f"  │ Buy signals profitable │  37/50   │  41/50   │ {buy_profitable:>2d}/{buy_eligible:<2d}    │")
    print(f"  │ Sharpe beats B&H       │   9/50   │  14/50   │ {sharpe_beat:>2d}/{sharpe_eligible:<2d}    │")
    print(f"  │ Median Sharpe          │  0.471   │  0.487   │ {safe_median(sharpes):>6.3f}   │")
    print("  └────────────────────────┴──────────┴──────────┴──────────┘")

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
