"""
CSI v6: Surgical Post-Processing Fix on Proven v3
===================================================
Strategy: Keep v3 CSI EXACTLY as-is (proven buy side 56.7% hit rate).
Apply 3 POST-PROCESSING corrections to fix sell signals:

1. OVERSOLD SUPPRESSION: When RSI < 35, pull negative CSI toward zero
   (prevents false sell signals at bottoms where bounces happen)

2. OVERBOUGHT BOOST: When RSI was >65 and is declining, push CSI more negative
   (improves sell signals at tops during exhaustion)

3. EXTENSION BOOST: When price is far above SMA50, push CSI more negative
   (mean reversion pressure from overextension)

Previous results:
  v2: Sell Hit 38.9%, Sell Avg +0.890%, Median Sharpe 0.471
  v3: Sell Hit 41.0%, Sell Avg +0.867%, Median Sharpe 0.487
  v4: Too few signals (1% freq), v5: Too many sells (2.5x scaling broke everything)
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


def compute_csi_v6(df):
    """v3 CSI with post-processing corrections for sell signals."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    n = len(close)

    if n < 60:
        return pd.Series(dtype=float)

    # ════════════════════════════════════════════════════════
    # ═══ v3 CSI (UNCHANGED) ════════════════════════════════
    # ════════════════════════════════════════════════════════

    # 1. TREND CONTEXT
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

    # 2. MULTI-TF MOMENTUM
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

    # 3. VOLUME FLOW
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

    # 4. OSCILLATOR SIGNALS (Asymmetric)
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

    rsi_declining = (rsi < rsi.shift(3)).astype(float)
    stoch_declining = (stoch_k < stoch_k.shift(3)).astype(float)
    rsi_sell = np.where((rsi > 70) & (rsi_declining > 0), (rsi - 70) / 30, 0.0)
    stoch_sell = np.where((stoch_k > 80) & (stoch_declining > 0), (stoch_k - 80) / 20, 0.0)
    osc_sell = pd.Series(0.55 * rsi_sell + 0.45 * stoch_sell, index=close.index).clip(0, 1)

    # 5. MACD + ADX
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

    # 6. VOLATILITY
    vol_pct = vol_20.rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
    vol_dampener = 1.0 - 0.3 * (vol_pct - 0.5).clip(0, 0.5)

    # MASTER BLEND (v3 exactly)
    buy_raw = (
        0.30 * trend_score.clip(0, None) +
        0.25 * mom_score.clip(0, None) +
        0.15 * osc_buy +
        0.15 * vol_flow.clip(0, None) +
        0.10 * ma_context.clip(0, None) +
        0.05 * mom_accel_n.clip(0, None)
    )

    sell_raw = (
        0.25 * (-trend_score).clip(0, None) +
        0.25 * (-mom_score).clip(0, None) +
        0.15 * osc_sell +
        0.15 * (-vol_flow).clip(0, None) +
        0.10 * (-ma_context).clip(0, None) +
        0.05 * (-mom_accel_n).clip(0, None) +
        0.05 * (1 - above_50) * (-trend_score).clip(0, None)
    )

    raw_csi = (buy_raw - sell_raw) * 100

    vol_confirm = (np.sign(raw_csi) * np.sign(vol_flow)).clip(0, 1)
    raw_csi = raw_csi * (0.80 + 0.20 * vol_confirm)
    raw_csi = raw_csi * vol_dampener

    buy_factors = pd.concat([
        (trend_score > 0.05).astype(float),
        (mom_score > 0.05).astype(float),
        (vol_flow > 0.05).astype(float),
        (ma_context > 0).astype(float),
    ], axis=1).sum(axis=1)
    sell_factors = pd.concat([
        (trend_score < -0.05).astype(float),
        (mom_score < -0.05).astype(float),
        (vol_flow < -0.05).astype(float),
        (ma_context < 0).astype(float),
    ], axis=1).sum(axis=1)
    buy_gate = (buy_factors >= 2).astype(float)
    sell_gate = (sell_factors >= 2).astype(float)
    gate = np.where(raw_csi > 0, buy_gate, np.where(raw_csi < 0, sell_gate, 0.5))
    gate = pd.Series(gate, index=close.index)
    raw_csi = raw_csi * (0.3 + 0.7 * gate)
    raw_csi = raw_csi.clip(-100, 100)

    # Adaptive smoothing
    ema_f = raw_csi.ewm(span=3, adjust=False).mean()
    ema_s = raw_csi.ewm(span=8, adjust=False).mean()
    sw = adx_regime.fillna(0.5)
    v3_csi = sw * ema_f + (1 - sw) * ema_s

    # ════════════════════════════════════════════════════════
    # ═══ POST-PROCESSING CORRECTIONS (v6 additions) ═══════
    # ════════════════════════════════════════════════════════

    csi = v3_csi.copy()

    # CORRECTION 1: Oversold suppression
    # When RSI < 40, pull NEGATIVE CSI toward zero proportionally
    # This prevents strong sell signals at bottoms where bounces happen
    oversold_strength = pd.Series(
        np.where(rsi < 40, (40 - rsi) / 40, 0.0),
        index=close.index
    ).clip(0, 1)
    # Also use stochastic as secondary oversold detector
    stoch_oversold = pd.Series(
        np.where(stoch_k < 30, (30 - stoch_k) / 30, 0.0),
        index=close.index
    ).clip(0, 1)
    oversold_combined = np.maximum(oversold_strength, stoch_oversold)

    # Pull negative CSI toward zero: stronger when more oversold
    # correction = oversold * |csi_negative| * strength_factor
    neg_csi = csi.clip(upper=0)  # only negative values
    correction_up = oversold_combined * (-neg_csi) * 1.2  # positive amount to add
    csi = csi + correction_up  # this only affects negative CSI

    # CORRECTION 2: Overbought exhaustion boost
    # When RSI was recently > 65 and is now declining, push CSI more negative
    rsi_recent_max = rsi.rolling(10).max()
    rsi_was_ob = (rsi_recent_max > 65).astype(float)
    rsi_falling = (rsi < rsi.shift(3)).astype(float)
    rsi_drop_pct = ((rsi_recent_max - rsi) / 30).clip(0, 1)
    ob_exhaust = rsi_was_ob * rsi_falling * rsi_drop_pct
    csi = csi - ob_exhaust * 25  # push CSI down by up to 25 points

    # CORRECTION 3: Extension above mean
    # When price is far above SMA50, push CSI more negative (reversion pressure)
    ext_above = (close / sma_50 - 1).clip(0, None)
    ext_rank = ext_above.rolling(252, min_periods=60).rank(pct=True).fillna(0)
    # Only activate above 70th percentile of historical extension
    ext_boost = ((ext_rank - 0.70) * 3.33).clip(0, 1)  # ramp from 0 at 70th to 1 at 100th
    csi = csi - ext_boost * 15  # push CSI down by up to 15 points

    # CORRECTION 4: Momentum exhaustion from positive
    # When momentum was strong and is now fading, add sell pressure
    mom_avg5 = mom_score.rolling(5).mean()
    mom_was_pos = (mom_avg5.shift(5) > 0.10).astype(float)
    mom_dropping = (mom_score < mom_avg5.shift(5) - 0.05).astype(float)
    mom_drop_mag = ((mom_avg5.shift(5) - mom_score) / 0.3).clip(0, 1)
    mom_exhaust_signal = mom_was_pos * mom_dropping * mom_drop_mag
    csi = csi - mom_exhaust_signal * 12  # push CSI down by up to 12 points

    # Re-clip and smooth the corrections
    csi = csi.clip(-100, 100)
    # Light smoothing of corrections (don't re-smooth too much, v3 already smoothed)
    csi = csi.ewm(span=2, adjust=False).mean()

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
    dir_correct_5d = ((np.sign(csi_v) == np.sign(fwd_5_v)) | (csi_v == 0)).mean()

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
        "dir_acc_5d": round(dir_correct_5d * 100, 1),
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
    print("=" * 115)
    print("  CSI v6 — SURGICAL POST-PROCESSING: Oversold Suppression + Overbought/Extension Boost")
    print("=" * 115)
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

        csi = compute_csi_v6(df_r)
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

    print(f"{'Symbol':<10} {'Corr5d':>7} {'DirAcc5':>8} {'BuyHit':>7} {'SellHit':>8} "
          f"{'BuyAvg5d':>9} {'SellAvg5d':>10} {'Sharpe':>7} {'B&H Sh':>7} {'BuyCnt':>7} {'SellCnt':>8} {'SigFrq':>7}")
    print("-" * 115)

    for r in sorted(results, key=lambda x: x.get("sharpe") or -999, reverse=True):
        corr5 = f"{r['corr_5d']:.4f}" if r['corr_5d'] is not None else "   N/A"
        buy_hr = f"{r['buy_hit_rate']:.1f}%" if r['buy_hit_rate'] is not None else "  N/A"
        sell_hr = f"{r['sell_hit_rate']:.1f}%" if r['sell_hit_rate'] is not None else "  N/A"
        buy_avg = f"{r['buy_avg_5d_pct']:+.3f}%" if r['buy_avg_5d_pct'] is not None else "    N/A"
        sell_avg = f"{r['sell_avg_5d_pct']:+.3f}%" if r['sell_avg_5d_pct'] is not None else "     N/A"
        sharpe = f"{r['sharpe']:.3f}" if r['sharpe'] is not None else "  N/A"
        bh_sh = f"{r['bh_sharpe']:.3f}" if r['bh_sharpe'] is not None else "  N/A"
        print(f"{r['symbol']:<10} {corr5:>7} {r['dir_acc_5d']:>7.1f}% {buy_hr:>7} {sell_hr:>8} "
              f"{buy_avg:>9} {sell_avg:>10} {sharpe:>7} {bh_sh:>7} {r['buy_count']:>7d} {r['sell_count']:>8d} {r['signal_freq_pct']:>6.1f}%")

    print()
    print("=" * 115)
    print("  AGGREGATE STATISTICS")
    print("=" * 115)

    def safe_mean(vals):
        clean = [v for v in vals if v is not None and not np.isnan(v)]
        return np.mean(clean) if clean else float("nan")

    def safe_median(vals):
        clean = [v for v in vals if v is not None and not np.isnan(v)]
        return np.median(clean) if clean else float("nan")

    n = len(results)
    buy_hits = [r["buy_hit_rate"] for r in results]
    sell_hits = [r["sell_hit_rate"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    bh_sharpes = [r["bh_sharpe"] for r in results]
    buy_avgs = [r["buy_avg_5d_pct"] for r in results]
    sell_avgs = [r["sell_avg_5d_pct"] for r in results]
    dir_acc_5d = [r["dir_acc_5d"] for r in results]
    sig_freqs = [r["signal_freq_pct"] for r in results]

    print(f"  Assets tested:           {n}")
    print(f"  Mean Corr (5d fwd):      {safe_mean([r['corr_5d'] for r in results]):.4f}")
    print(f"  Mean Dir Accuracy (5d):  {safe_mean(dir_acc_5d):.1f}%")
    print(f"  Mean Buy Hit Rate:       {safe_mean(buy_hits):.1f}%")
    print(f"  Mean Sell Hit Rate:      {safe_mean(sell_hits):.1f}%")
    print(f"  Median Sharpe:           {safe_median(sharpes):.3f}")
    print(f"  Mean Sharpe:             {safe_mean(sharpes):.3f}")
    print(f"  Median B&H Sharpe:       {safe_median(bh_sharpes):.3f}")
    print(f"  Avg Buy Signal 5d Ret:   {safe_mean(buy_avgs):.3f}%")
    print(f"  Avg Sell Signal 5d Ret:  {safe_mean(sell_avgs):.3f}%")
    print(f"  Avg Signal Frequency:    {safe_mean(sig_freqs):.1f}%")

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
    print("  │ Metric                 │    v2    │    v3    │  v6 NOW  │")
    print("  ├────────────────────────┼──────────┼──────────┼──────────┤")
    print(f"  │ Mean Sell Hit Rate     │  38.9%   │  41.0%   │ {safe_mean(sell_hits):>5.1f}%   │")
    print(f"  │ Sell signals correct   │  13/50   │  11/50   │ {sell_correct:>2d}/{sell_eligible:<2d}    │")
    print(f"  │ Avg Sell 5d Ret        │ +0.890%  │ +0.867%  │{safe_mean(sell_avgs):>+7.3f}%  │")
    print(f"  │ Mean Buy Hit Rate      │  56.1%   │  56.7%   │ {safe_mean(buy_hits):>5.1f}%   │")
    print(f"  │ Buy signals profitable │  37/50   │  41/50   │ {buy_profitable:>2d}/{buy_eligible:<2d}    │")
    print(f"  │ Sharpe beats B&H       │   9/50   │  14/50   │ {sharpe_beat:>2d}/{sharpe_eligible:<2d}    │")
    print(f"  │ Median Sharpe          │  0.471   │  0.487   │ {safe_median(sharpes):>6.3f}   │")
    print(f"  │ Signal Frequency       │  17.0%   │  17.0%   │ {safe_mean(sig_freqs):>5.1f}%   │")
    print("  └────────────────────────┴──────────┴──────────┴──────────┘")

    print()
    print("=" * 115)


if __name__ == "__main__":
    main()
