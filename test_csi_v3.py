"""
CSI v3 Comprehensive Test Harness
=================================
Tests the Composite Signal Index on 53 assets (50 stocks + GLD + SLV + BTC-USD).

Metrics:
  1. Forward return correlation (does CSI predict next 5d/10d returns?)
  2. Extreme signal accuracy (when CSI > 50 or < -50, is it right?)
  3. Hit rate at thresholds (+30 buy, -30 sell)
  4. Avg forward return when signal fires
  5. Sharpe of signal-following strategy
  6. Max drawdown of signal-following strategy
  7. Signal quality score (composite metric)
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

# Test universe: 50 diversified stocks + gold + silver + bitcoin
TEST_UNIVERSE = [
    # Tech (12)
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "CRM", "ADBE", "NFLX", "CRWD",
    # Finance (6)
    "JPM", "BAC", "GS", "MS", "SCHW", "AFRM",
    # Healthcare (5)
    "JNJ", "UNH", "PFE", "ABBV", "MRNA",
    # Defence (4)
    "LMT", "RTX", "NOC", "GD",
    # Industrials (5)
    "CAT", "DE", "BA", "UPS", "GE",
    # Energy (4)
    "XOM", "CVX", "COP", "SLB",
    # Consumer (6)
    "HD", "NKE", "SBUX", "PG", "KO", "COST",
    # Small/Mid (4)
    "UPST", "IONQ", "DKNG", "SNAP",
    # Index ETFs (4)
    "SPY", "QQQ", "IWM", "DIA",
    # Commodities + Crypto (3)
    "GLD", "SLV", "BTC-USD",
]

PRICES_DIR = os.path.join(SRC_DIR, "data", "prices")


def load_ohlcv(symbol: str) -> pd.DataFrame:
    """Load OHLCV data from CSV."""
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


def compute_csi_v3(df: pd.DataFrame) -> pd.Series:
    """Compute CSI v3 matching chart_service.py logic exactly."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    n = len(close)

    if n < 50:
        return pd.Series(dtype=float)

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
    rel_vol = (volume / vol_sma20).clip(0.1, 5.0)

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
    vol_60 = ret_1.rolling(60).std()
    vol_pct = vol_20.rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
    vol_dampener = 1.0 - 0.3 * (vol_pct - 0.5).clip(0, 0.5)

    # MASTER BLEND
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

    # Signal gating
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
    csi = sw * ema_f + (1 - sw) * ema_s

    return csi


def evaluate_csi(symbol: str, csi: pd.Series, close: pd.Series) -> dict:
    """Compute performance metrics for CSI on a single asset."""
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
    strong_buy_count = strong_buy.sum()
    strong_sell_count = strong_sell.sum()

    buy_avg_5d = fwd_5_v[buy_mask].mean() * 100 if buy_count > 5 else np.nan
    sell_avg_5d = fwd_5_v[sell_mask].mean() * 100 if sell_count > 5 else np.nan
    strong_buy_avg = fwd_5_v[strong_buy].mean() * 100 if strong_buy_count > 3 else np.nan
    strong_sell_avg = fwd_5_v[strong_sell].mean() * 100 if strong_sell_count > 3 else np.nan

    buy_hit = (fwd_5_v[buy_mask] > 0).mean() if buy_count > 5 else np.nan
    sell_hit = (fwd_5_v[sell_mask] < 0).mean() if sell_count > 5 else np.nan

    position = np.sign(csi_v).clip(0, 1)
    strat_ret = position * fwd_1_v
    strat_ret = strat_ret.dropna()
    buy_hold_ret = fwd_1_v.dropna()

    if len(strat_ret) > 20 and strat_ret.std() > 0:
        sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
    else:
        sharpe = np.nan

    bh_sharpe = (buy_hold_ret.mean() / buy_hold_ret.std()) * np.sqrt(252) if buy_hold_ret.std() > 0 else np.nan

    cumret = (1 + strat_ret).cumprod()
    peak = cumret.cummax()
    dd = (cumret - peak) / peak
    max_dd = dd.min() * 100

    if len(strat_ret) > 0:
        total_ret = cumret.iloc[-1] / cumret.iloc[0] if cumret.iloc[0] > 0 else 1
        years = len(strat_ret) / 252
        cagr = (total_ret ** (1 / max(years, 0.1)) - 1) * 100
    else:
        cagr = np.nan

    signal_pct = ((csi_v.abs() > 30).sum() / len(csi_v)) * 100

    return {
        "symbol": symbol,
        "n_bars": len(csi_v),
        "corr_5d": round(corr_5d, 4) if not np.isnan(corr_5d) else None,
        "corr_10d": round(corr_10d, 4) if not np.isnan(corr_10d) else None,
        "dir_acc_5d": round(dir_correct_5d * 100, 1),
        "dir_acc_10d": round(dir_correct_10d * 100, 1),
        "buy_count": int(buy_count),
        "sell_count": int(sell_count),
        "buy_avg_5d_pct": round(buy_avg_5d, 3) if not np.isnan(buy_avg_5d) else None,
        "sell_avg_5d_pct": round(sell_avg_5d, 3) if not np.isnan(sell_avg_5d) else None,
        "strong_buy_avg": round(strong_buy_avg, 3) if not np.isnan(strong_buy_avg) else None,
        "strong_sell_avg": round(strong_sell_avg, 3) if not np.isnan(strong_sell_avg) else None,
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
    print("  CSI v3 COMPREHENSIVE TEST — Asymmetric Multi-Factor Engine")
    print("  Universe: 50 stocks + GLD + SLV + BTC-USD")
    print("=" * 100)
    print()

    results = []
    skipped = []

    for sym in TEST_UNIVERSE:
        df = load_ohlcv(sym)
        if df.empty or len(df) < 100:
            skipped.append(sym)
            continue

        close = df["close"].astype(float).reset_index(drop=True)
        df_reset = df.reset_index(drop=True)
        df_reset["close"] = close
        df_reset["high"] = df["high"].astype(float).reset_index(drop=True)
        df_reset["low"] = df["low"].astype(float).reset_index(drop=True)
        df_reset["volume"] = df["volume"].astype(float).reset_index(drop=True)

        csi = compute_csi_v3(df_reset)
        if csi.empty:
            skipped.append(sym)
            continue

        metrics = evaluate_csi(sym, csi, close)
        if "error" in metrics:
            skipped.append(sym)
            continue

        results.append(metrics)

    if skipped:
        print(f"  Skipped ({len(skipped)}): {', '.join(skipped)}")
        print()

    print(f"{'Symbol':<10} {'Corr5d':>7} {'Corr10d':>8} {'DirAcc5':>8} {'BuyHit':>7} {'SellHit':>8} "
          f"{'Sharpe':>7} {'B&H Sh':>7} {'CAGR':>6} {'MaxDD':>6} {'SigFreq':>8}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: x.get("sharpe") or -999, reverse=True):
        corr5 = f"{r['corr_5d']:.4f}" if r['corr_5d'] is not None else "   N/A"
        corr10 = f"{r['corr_10d']:.4f}" if r['corr_10d'] is not None else "   N/A"
        buy_hr = f"{r['buy_hit_rate']:.1f}%" if r['buy_hit_rate'] is not None else "  N/A"
        sell_hr = f"{r['sell_hit_rate']:.1f}%" if r['sell_hit_rate'] is not None else "  N/A"
        sharpe = f"{r['sharpe']:.3f}" if r['sharpe'] is not None else "  N/A"
        bh_sh = f"{r['bh_sharpe']:.3f}" if r['bh_sharpe'] is not None else "  N/A"
        cagr = f"{r['cagr_pct']:.1f}%" if r['cagr_pct'] is not None else " N/A"
        print(f"{r['symbol']:<10} {corr5:>7} {corr10:>8} {r['dir_acc_5d']:>7.1f}% {buy_hr:>7} {sell_hr:>8} "
              f"{sharpe:>7} {bh_sh:>7} {cagr:>6} {r['max_dd_pct']:>5.1f}% {r['signal_freq_pct']:>7.1f}%")

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
    corr10_vals = [r["corr_10d"] for r in results]
    dir_acc_5d = [r["dir_acc_5d"] for r in results]
    buy_hits = [r["buy_hit_rate"] for r in results]
    sell_hits = [r["sell_hit_rate"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    bh_sharpes = [r["bh_sharpe"] for r in results]
    cagrs = [r["cagr_pct"] for r in results]
    max_dds = [r["max_dd_pct"] for r in results]
    buy_avgs = [r["buy_avg_5d_pct"] for r in results]
    sell_avgs = [r["sell_avg_5d_pct"] for r in results]

    print(f"  Assets tested:           {n}")
    print(f"  Mean Corr (5d fwd):      {safe_mean(corr5_vals):.4f}")
    print(f"  Mean Corr (10d fwd):     {safe_mean(corr10_vals):.4f}")
    print(f"  Mean Dir Accuracy (5d):  {safe_mean(dir_acc_5d):.1f}%")
    print(f"  Mean Buy Hit Rate:       {safe_mean(buy_hits):.1f}%")
    print(f"  Mean Sell Hit Rate:      {safe_mean(sell_hits):.1f}%")
    print(f"  Median Sharpe:           {safe_median(sharpes):.3f}")
    print(f"  Mean Sharpe:             {safe_mean(sharpes):.3f}")
    print(f"  Median B&H Sharpe:       {safe_median(bh_sharpes):.3f}")
    print(f"  Mean CAGR:               {safe_mean(cagrs):.1f}%")
    print(f"  Mean Max Drawdown:       {safe_mean(max_dds):.1f}%")
    print(f"  Avg Buy Signal 5d Ret:   {safe_mean(buy_avgs):.3f}%")
    print(f"  Avg Sell Signal 5d Ret:  {safe_mean(sell_avgs):.3f}%")
    print()

    sharpe_beat = sum(1 for r in results
                      if r["sharpe"] is not None and r["bh_sharpe"] is not None
                      and r["sharpe"] > r["bh_sharpe"])
    sharpe_eligible = sum(1 for r in results
                          if r["sharpe"] is not None and r["bh_sharpe"] is not None)
    buy_profitable = sum(1 for r in results if r["buy_avg_5d_pct"] is not None and r["buy_avg_5d_pct"] > 0)
    buy_eligible = sum(1 for r in results if r["buy_avg_5d_pct"] is not None)
    sell_profitable = sum(1 for r in results if r["sell_avg_5d_pct"] is not None and r["sell_avg_5d_pct"] < 0)
    sell_eligible = sum(1 for r in results if r["sell_avg_5d_pct"] is not None)

    print("  QUALITY BENCHMARKS:")
    print(f"    Sharpe beats B&H:      {sharpe_beat}/{sharpe_eligible} ({100*sharpe_beat/max(sharpe_eligible,1):.0f}%)")
    print(f"    Buy signals profitable:{buy_profitable}/{buy_eligible} ({100*buy_profitable/max(buy_eligible,1):.0f}%)")
    print(f"    Sell signals correct:  {sell_profitable}/{sell_eligible} ({100*sell_profitable/max(sell_eligible,1):.0f}%)")
    print(f"    Dir accuracy > 50%:    {sum(1 for d in dir_acc_5d if d > 50)}/{n}")

    print()
    print("  STRONG SIGNAL ANALYSIS (|CSI| > 50):")
    sb_avgs = [r["strong_buy_avg"] for r in results if r["strong_buy_avg"] is not None]
    ss_avgs = [r["strong_sell_avg"] for r in results if r["strong_sell_avg"] is not None]
    print(f"    Strong Buy avg 5d ret: {np.mean(sb_avgs):.3f}% ({len(sb_avgs)} assets)" if sb_avgs else "    Strong Buy: insufficient data")
    print(f"    Strong Sell avg 5d ret:{np.mean(ss_avgs):.3f}% ({len(ss_avgs)} assets)" if ss_avgs else "    Strong Sell: insufficient data")

    # v2 baseline comparison
    print()
    print("  v2 BASELINE (from previous test):")
    print("    Mean Sell Hit Rate:    38.9%")
    print("    Sell signals correct:  13/50 (26%)")
    print("    Avg Sell 5d Ret:       +0.890% (WRONG direction)")
    print("    Buy signals profitable:37/50 (74%)")
    print("    Mean Buy Hit Rate:     56.1%")
    print("    Sharpe beats B&H:      9/50 (18%)")

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
