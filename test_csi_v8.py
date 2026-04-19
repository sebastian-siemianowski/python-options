"""
CSI v8: v3 + CONDITIONAL Corrections Based on Trend Context
=============================================================
Key insight from v7: indiscriminate bullish bias and oversold flip don't work.
Need CONDITIONAL logic:

1. OVERSOLD in UPTREND (close > SMA200): FLIP CSI to positive (buying opportunity)
   OVERSOLD in DOWNTREND (close < SMA200): KEEP CSI negative (genuine bear)

2. OVERBOUGHT with ROLLING OVER (SMA50 slope negative): STRENGTHEN sell
   OVERBOUGHT with STRONG TREND (SMA50 slope positive): KEEP CSI positive

3. VOLUME DIVERGENCE: Price making new highs but volume declining → sell pressure

No blanket bullish bias. The trend context determines correction direction.
"""

import os, sys, warnings
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


def compute_csi_v8(df):
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    n = len(close)
    if n < 60:
        return pd.Series(dtype=float)

    # ═══ v3 CSI CORE ══════════════════════════════════════
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200, min_periods=60).mean()
    above_20 = (close > sma_20).astype(float)
    above_50 = (close > sma_50).astype(float)
    above_200 = (close > sma_200).astype(float)

    ma20_slope = sma_20.pct_change(5)
    ma20_slope_n = (ma20_slope / ma20_slope.abs().rolling(60).max().replace(0, float("nan"))).clip(-1, 1)
    ma50_slope = sma_50.pct_change(10)
    ma50_slope_n = (ma50_slope / ma50_slope.abs().rolling(60).max().replace(0, float("nan"))).clip(-1, 1)
    ma200_slope = sma_200.pct_change(20)
    ma200_slope_positive = (ma200_slope > 0).astype(float)

    ma_context = (0.25 * (above_20 * 2 - 1) + 0.25 * (above_50 * 2 - 1) +
                  0.25 * ma20_slope_n + 0.25 * ma50_slope_n)

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

    up_vol = volume.where(close > close.shift(1), 0.0).rolling(10).sum()
    dn_vol = volume.where(close <= close.shift(1), 0.0).rolling(10).sum()
    vol_ratio = ((up_vol - dn_vol) / (up_vol + dn_vol).replace(0, float("nan"))).clip(-1, 1)
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

    vol_pct = vol_20.rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
    vol_dampener = 1.0 - 0.3 * (vol_pct - 0.5).clip(0, 0.5)

    buy_raw = (
        0.30 * trend_score.clip(0, None) + 0.25 * mom_score.clip(0, None) +
        0.15 * osc_buy + 0.15 * vol_flow.clip(0, None) +
        0.10 * ma_context.clip(0, None) + 0.05 * mom_accel_n.clip(0, None)
    )
    sell_raw = (
        0.25 * (-trend_score).clip(0, None) + 0.25 * (-mom_score).clip(0, None) +
        0.15 * osc_sell + 0.15 * (-vol_flow).clip(0, None) +
        0.10 * (-ma_context).clip(0, None) + 0.05 * (-mom_accel_n).clip(0, None) +
        0.05 * (1 - above_50) * (-trend_score).clip(0, None)
    )

    raw_csi = (buy_raw - sell_raw) * 100
    vol_confirm = (np.sign(raw_csi) * np.sign(vol_flow)).clip(0, 1)
    raw_csi = raw_csi * (0.80 + 0.20 * vol_confirm)
    raw_csi = raw_csi * vol_dampener

    buy_factors = pd.concat([
        (trend_score > 0.05).astype(float), (mom_score > 0.05).astype(float),
        (vol_flow > 0.05).astype(float), (ma_context > 0).astype(float),
    ], axis=1).sum(axis=1)
    sell_factors = pd.concat([
        (trend_score < -0.05).astype(float), (mom_score < -0.05).astype(float),
        (vol_flow < -0.05).astype(float), (ma_context < 0).astype(float),
    ], axis=1).sum(axis=1)
    buy_gate = (buy_factors >= 2).astype(float)
    sell_gate = (sell_factors >= 2).astype(float)
    gate = np.where(raw_csi > 0, buy_gate, np.where(raw_csi < 0, sell_gate, 0.5))
    gate = pd.Series(gate, index=close.index)
    raw_csi = raw_csi * (0.3 + 0.7 * gate)
    raw_csi = raw_csi.clip(-100, 100)

    ema_f = raw_csi.ewm(span=3, adjust=False).mean()
    ema_s = raw_csi.ewm(span=8, adjust=False).mean()
    sw = adx_regime.fillna(0.5)
    v3_csi = sw * ema_f + (1 - sw) * ema_s

    # ═══ v8 CONDITIONAL CORRECTIONS ═══════════════════════

    csi = v3_csi.copy()

    # CORRECTION 1: OVERSOLD in UPTREND → Push CSI positive (buying opportunity)
    # OVERSOLD in DOWNTREND → Keep CSI negative (genuine bear)
    oversold = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    stoch_os = pd.Series(np.where(stoch_k < 20, (20 - stoch_k) / 20, 0.0), index=close.index).clip(0, 1)
    os_strength = np.maximum(oversold, stoch_os)

    # In uptrend: push CSI positive (oversold bounce)
    uptrend = above_200.fillna(0)
    uptrend_os_boost = os_strength * uptrend * 40  # up to +40 in uptrend
    csi = csi + uptrend_os_boost

    # In downtrend: STRENGTHEN the sell signal (genuine bear, not a buying opportunity)
    downtrend = (1 - above_200).fillna(0)
    downtrend_os_penalty = os_strength * downtrend * 10  # up to -10 more in downtrend
    csi = csi - downtrend_os_penalty

    # CORRECTION 2: OVERBOUGHT with trend ROLLING OVER → Sell pressure
    # Only trigger when SMA50 slope is turning negative (exhaustion confirmed)
    rsi_ob = pd.Series(np.where(rsi > 68, (rsi - 68) / 32, 0.0), index=close.index).clip(0, 1)
    ma50_rolling_over = (ma50_slope < 0).astype(float)
    ob_with_rollover = rsi_ob * ma50_rolling_over * 25
    csi = csi - ob_with_rollover

    # OVERBOUGHT with declining RSI (even if trend still up) → lighter sell pressure
    rsi_max10 = rsi.rolling(10).max()
    rsi_was_ob = (rsi_max10 > 70).astype(float)
    rsi_declining_fast = ((rsi_max10 - rsi) / 20).clip(0, 1)
    ob_exhaustion = rsi_was_ob * rsi_declining_fast * 12
    csi = csi - ob_exhaustion

    # CORRECTION 3: VOLUME DIVERGENCE → Price at highs but volume declining (distribution)
    price_near_high = ((close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min()).replace(0, float("nan")))
    price_near_high = price_near_high.clip(0, 1)
    vol_10 = volume.rolling(10).mean()
    vol_30 = volume.rolling(30).mean()
    vol_declining = ((vol_30 - vol_10) / vol_30.replace(0, float("nan"))).clip(0, 1)
    # Only trigger when price is in upper 30% of 20-day range AND volume is declining
    vol_div = (price_near_high > 0.70).astype(float) * vol_declining * 8
    csi = csi - vol_div

    # CORRECTION 4: BREAKDOWN ACCELERATOR
    # When price drops below SMA50 AND volume is INCREASING → amplify sell signal
    below_50 = (1 - above_50).astype(float)
    vol_increasing = (vol_10 > vol_30 * 1.1).astype(float)
    neg_csi = (csi < 0).astype(float)
    breakdown = below_50 * vol_increasing * neg_csi * 8
    csi = csi - breakdown

    csi = csi.clip(-100, 100)
    # Very light smoothing (v3 already smoothed)
    csi = csi.ewm(span=2, adjust=False).mean()

    return csi


def evaluate_csi(symbol, csi, close):
    valid = csi.notna() & close.notna()
    csi = csi[valid]
    close = close[valid]
    if len(csi) < 60:
        return {"symbol": symbol, "error": "insufficient data"}

    fwd_5 = close.pct_change(5).shift(-5)
    fwd_1 = close.pct_change(1).shift(-1)
    fwd_10 = close.pct_change(10).shift(-10)
    mask = fwd_10.notna() & csi.notna()
    csi_v = csi[mask]; fwd_5_v = fwd_5[mask]; fwd_1_v = fwd_1[mask]
    if len(csi_v) < 30:
        return {"symbol": symbol, "error": "insufficient valid data"}

    corr_5d = csi_v.corr(fwd_5_v)
    buy_mask = csi_v > 30; sell_mask = csi_v < -30
    strong_buy = csi_v > 50; strong_sell = csi_v < -50
    bc = buy_mask.sum(); sc = sell_mask.sum()

    buy_avg = fwd_5_v[buy_mask].mean() * 100 if bc > 5 else np.nan
    sell_avg = fwd_5_v[sell_mask].mean() * 100 if sc > 5 else np.nan
    sb_avg = fwd_5_v[strong_buy].mean() * 100 if strong_buy.sum() > 3 else np.nan
    ss_avg = fwd_5_v[strong_sell].mean() * 100 if strong_sell.sum() > 3 else np.nan
    buy_hit = (fwd_5_v[buy_mask] > 0).mean() if bc > 5 else np.nan
    sell_hit = (fwd_5_v[sell_mask] < 0).mean() if sc > 5 else np.nan

    position = np.sign(csi_v).clip(0, 1)
    strat_ret = (position * fwd_1_v).dropna()
    bh_ret = fwd_1_v.dropna()
    sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) if len(strat_ret) > 20 and strat_ret.std() > 0 else np.nan
    bh_sharpe = (bh_ret.mean() / bh_ret.std()) * np.sqrt(252) if bh_ret.std() > 0 else np.nan

    cumret = (1 + strat_ret).cumprod()
    max_dd = ((cumret - cumret.cummax()) / cumret.cummax()).min() * 100

    signal_pct = ((csi_v.abs() > 30).sum() / len(csi_v)) * 100
    pct_pos = (csi_v > 0).mean() * 100
    pos_ret = fwd_1_v[csi_v > 0].mean() * 252 * 100 if (csi_v > 0).sum() > 20 else np.nan
    neg_ret = fwd_1_v[csi_v <= 0].mean() * 252 * 100 if (csi_v <= 0).sum() > 20 else np.nan

    return {
        "symbol": symbol, "n_bars": len(csi_v),
        "corr_5d": round(corr_5d, 4) if not np.isnan(corr_5d) else None,
        "buy_count": int(bc), "sell_count": int(sc),
        "buy_avg_5d": round(buy_avg, 3) if not np.isnan(buy_avg) else None,
        "sell_avg_5d": round(sell_avg, 3) if not np.isnan(sell_avg) else None,
        "sb_avg": round(sb_avg, 3) if not np.isnan(sb_avg) else None,
        "ss_avg": round(ss_avg, 3) if not np.isnan(ss_avg) else None,
        "buy_hit": round(buy_hit * 100, 1) if buy_hit is not None and not np.isnan(buy_hit) else None,
        "sell_hit": round(sell_hit * 100, 1) if sell_hit is not None and not np.isnan(sell_hit) else None,
        "sharpe": round(sharpe, 3) if not np.isnan(sharpe) else None,
        "bh_sharpe": round(bh_sharpe, 3) if not np.isnan(bh_sharpe) else None,
        "max_dd": round(max_dd, 1),
        "sig_freq": round(signal_pct, 1),
        "pct_pos": round(pct_pos, 1),
        "pos_ret": round(pos_ret, 1) if not np.isnan(pos_ret) else None,
        "neg_ret": round(neg_ret, 1) if not np.isnan(neg_ret) else None,
    }


def main():
    print("=" * 125)
    print("  CSI v8 — CONDITIONAL: Uptrend Oversold Flip + Rollover Sell + Volume Divergence + Breakdown Accelerator")
    print("=" * 125)

    results = []; skipped = []
    for sym in TEST_UNIVERSE:
        df = load_ohlcv(sym)
        if df.empty or len(df) < 100:
            skipped.append(sym); continue
        df_r = df.reset_index(drop=True)
        for col in ["close", "high", "low", "volume"]:
            df_r[col] = df[col].astype(float).reset_index(drop=True)
        csi = compute_csi_v8(df_r)
        if csi.empty:
            skipped.append(sym); continue
        m = evaluate_csi(sym, csi, df_r["close"])
        if "error" in m:
            skipped.append(sym); continue
        results.append(m)

    if skipped:
        print(f"\n  Skipped ({len(skipped)}): {', '.join(skipped)}")

    print(f"\n{'Sym':<8} {'BuyHit':>7} {'SellHit':>8} {'BuyAvg5':>8} {'SellAvg5':>9} "
          f"{'Sharpe':>7} {'B&H':>6} {'CSI>0':>6} {'PosRet':>7} {'NegRet':>7} {'SigFrq':>7} {'MaxDD':>7}")
    print("-" * 125)

    for r in sorted(results, key=lambda x: x.get("sharpe") or -999, reverse=True):
        bhr = f"{r['buy_hit']:.0f}%" if r['buy_hit'] else " N/A"
        shr = f"{r['sell_hit']:.0f}%" if r['sell_hit'] else " N/A"
        ba = f"{r['buy_avg_5d']:+.2f}%" if r['buy_avg_5d'] is not None else "   N/A"
        sa = f"{r['sell_avg_5d']:+.2f}%" if r['sell_avg_5d'] is not None else "    N/A"
        sh = f"{r['sharpe']:.3f}" if r['sharpe'] else " N/A"
        bh = f"{r['bh_sharpe']:.3f}" if r['bh_sharpe'] else " N/A"
        pr = f"{r['pos_ret']:+.0f}%" if r['pos_ret'] is not None else "  N/A"
        nr = f"{r['neg_ret']:+.0f}%" if r['neg_ret'] is not None else "  N/A"
        print(f"{r['symbol']:<8} {bhr:>7} {shr:>8} {ba:>8} {sa:>9} "
              f"{sh:>7} {bh:>6} {r['pct_pos']:>5.0f}% {pr:>7} {nr:>7} {r['sig_freq']:>6.1f}% {r['max_dd']:>6.1f}%")

    print()
    print("=" * 125)
    print("  AGGREGATE")
    print("=" * 125)

    def sm(vals):
        c = [v for v in vals if v is not None and not np.isnan(v)]
        return np.mean(c) if c else float("nan")
    def smed(vals):
        c = [v for v in vals if v is not None and not np.isnan(v)]
        return np.median(c) if c else float("nan")

    n = len(results)
    buy_hits = [r["buy_hit"] for r in results]
    sell_hits = [r["sell_hit"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    bh_sharpes = [r["bh_sharpe"] for r in results]
    buy_avgs = [r["buy_avg_5d"] for r in results]
    sell_avgs = [r["sell_avg_5d"] for r in results]
    pct_pos = [r["pct_pos"] for r in results]
    pos_rets = [r["pos_ret"] for r in results]
    neg_rets = [r["neg_ret"] for r in results]

    print(f"  Assets tested:           {n}")
    print(f"  Mean Buy Hit Rate:       {sm(buy_hits):.1f}%")
    print(f"  Mean Sell Hit Rate:      {sm(sell_hits):.1f}%")
    print(f"  Median Sharpe:           {smed(sharpes):.3f}")
    print(f"  Mean Sharpe:             {sm(sharpes):.3f}")
    print(f"  Median B&H Sharpe:       {smed(bh_sharpes):.3f}")
    print(f"  Avg Buy 5d Ret:          {sm(buy_avgs):.3f}%")
    print(f"  Avg Sell 5d Ret:         {sm(sell_avgs):.3f}%")

    print(f"\n  REGIME SEPARATION:")
    print(f"    Avg % time CSI > 0:    {sm(pct_pos):.0f}%")
    print(f"    Avg ann ret CSI > 0:   {sm(pos_rets):.1f}%")
    print(f"    Avg ann ret CSI <= 0:  {sm(neg_rets):.1f}%")
    spread = sm(pos_rets) - sm(neg_rets)
    print(f"    Regime spread:         {spread:+.1f}%")

    sharpe_beat = sum(1 for r in results if r["sharpe"] is not None and r["bh_sharpe"] is not None and r["sharpe"] > r["bh_sharpe"])
    sharpe_elig = sum(1 for r in results if r["sharpe"] is not None and r["bh_sharpe"] is not None)
    buy_prof = sum(1 for r in results if r["buy_avg_5d"] is not None and r["buy_avg_5d"] > 0)
    buy_elig = sum(1 for r in results if r["buy_avg_5d"] is not None)
    sell_corr = sum(1 for r in results if r["sell_avg_5d"] is not None and r["sell_avg_5d"] < 0)
    sell_elig = sum(1 for r in results if r["sell_avg_5d"] is not None)

    print(f"\n  QUALITY:")
    print(f"    Sharpe beats B&H:       {sharpe_beat}/{sharpe_elig} ({100*sharpe_beat/max(sharpe_elig,1):.0f}%)")
    print(f"    Buy profitable:         {buy_prof}/{buy_elig} ({100*buy_prof/max(buy_elig,1):.0f}%)")
    print(f"    Sell correct:           {sell_corr}/{sell_elig} ({100*sell_corr/max(sell_elig,1):.0f}%)")

    # Regime separation quality: how many assets have pos_ret > neg_ret
    good_sep = sum(1 for r in results if r["pos_ret"] is not None and r["neg_ret"] is not None and r["pos_ret"] > r["neg_ret"])
    sep_elig = sum(1 for r in results if r["pos_ret"] is not None and r["neg_ret"] is not None)
    print(f"    Good regime separation: {good_sep}/{sep_elig} ({100*good_sep/max(sep_elig,1):.0f}%)")

    sb = [r["sb_avg"] for r in results if r["sb_avg"] is not None]
    ss = [r["ss_avg"] for r in results if r["ss_avg"] is not None]
    print(f"\n  STRONG SIGNALS (|CSI| > 50):")
    if sb: print(f"    Strong Buy avg 5d:  {np.mean(sb):.3f}% ({len(sb)} assets)")
    if ss: print(f"    Strong Sell avg 5d: {np.mean(ss):.3f}% ({len(ss)} assets)")

    print(f"\n  ┌────────────────────────┬──────────┬──────────┬──────────┬──────────┐")
    print(f"  │ Metric                 │    v2    │    v3    │    v7    │  v8 NOW  │")
    print(f"  ├────────────────────────┼──────────┼──────────┼──────────┼──────────┤")
    print(f"  │ Mean Sell Hit Rate     │  38.9%   │  41.0%   │  38.9%   │ {sm(sell_hits):>5.1f}%   │")
    print(f"  │ Sell correct           │  13/50   │  11/50   │  10/32   │ {sell_corr:>2d}/{sell_elig:<2d}    │")
    print(f"  │ Avg Sell 5d Ret        │ +0.890%  │ +0.867%  │ +0.646%  │{sm(sell_avgs):>+7.3f}%  │")
    print(f"  │ Mean Buy Hit Rate      │  56.1%   │  56.7%   │  56.3%   │ {sm(buy_hits):>5.1f}%   │")
    print(f"  │ Buy profitable         │  37/50   │  41/50   │  39/50   │ {buy_prof:>2d}/{buy_elig:<2d}    │")
    print(f"  │ Sharpe beats B&H       │   9/50   │  14/50   │  14/50   │ {sharpe_beat:>2d}/{sharpe_elig:<2d}    │")
    print(f"  │ Median Sharpe          │  0.471   │  0.487   │  0.585   │ {smed(sharpes):>6.3f}   │")
    print(f"  │ Regime spread          │   N/A    │   N/A    │  -5.2%   │{spread:>+6.1f}%   │")
    print(f"  └────────────────────────┴──────────┴──────────┴──────────┴──────────┘")
    print()


if __name__ == "__main__":
    main()
