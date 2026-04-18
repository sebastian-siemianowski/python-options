"""
CSI v2 Comprehensive Test Harness
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
    # Try various filename patterns
    for suffix in ["_1d.csv", ".csv"]:
        path = os.path.join(PRICES_DIR, f"{symbol}{suffix}")
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Normalize column names
            df.columns = [c.lower().strip() for c in df.columns]
            # Find date column
            for dc in ["date", "datetime", "timestamp"]:
                if dc in df.columns:
                    df["date"] = pd.to_datetime(df[dc]).dt.strftime("%Y-%m-%d")
                    break
            # Ensure numeric
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["close", "high", "low", "volume"])
            return df
    return pd.DataFrame()


def compute_csi(df: pd.DataFrame) -> pd.Series:
    """Compute CSI v2 matching chart_service.py logic, return aligned Series."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    n = len(close)

    if n < 50:
        return pd.Series(dtype=float)

    nan_s = pd.Series(np.nan, index=close.index)

    # LAYER 1: Multi-Timeframe Momentum
    ret_1 = close.pct_change(1)
    ret_5 = close.pct_change(5)
    ret_14 = close.pct_change(14)
    ret_30 = close.pct_change(30)

    vol_20 = ret_1.rolling(20).std().replace(0, float("nan"))

    mom_fast = (ret_5 / (vol_20 * np.sqrt(5))).clip(-3, 3) / 3
    mom_med = (ret_14 / (vol_20 * np.sqrt(14))).clip(-3, 3) / 3
    mom_slow = (ret_30 / (vol_20 * np.sqrt(30))).clip(-3, 3) / 3

    raw_mom = 0.50 * mom_fast + 0.30 * mom_med + 0.20 * mom_slow
    signs = pd.concat([np.sign(mom_fast), np.sign(mom_med), np.sign(mom_slow)], axis=1)
    agreement = signs.sum(axis=1).abs() / 3.0
    mom_score = raw_mom * (0.6 + 0.4 * agreement)

    # LAYER 2: Volatility Regime
    vol_60 = ret_1.rolling(60).std()
    vol_percentile = vol_20.rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
    vol_ratio = vol_20 / vol_60.replace(0, float("nan"))
    vol_compression = (1.0 - vol_ratio.clip(0.3, 2.0)).clip(-0.5, 0.5)
    vol_dampener = 1.0 - 0.4 * (vol_percentile - 0.5).clip(0, 0.5)

    # LAYER 3: Volume Confirmation
    vol_sma20 = volume.rolling(20).mean().replace(0, float("nan"))
    rel_vol = (volume / vol_sma20).clip(0.1, 5.0)
    price_dir = np.sign(close.diff())
    vol_surge = (rel_vol - 1.0).clip(-1, 3)
    pv_align = (price_dir * vol_surge * 0.3).clip(-1, 1)

    clv = ((close - low) - (high - close)) / (high - low).replace(0, float("nan"))
    ad_line = (clv * volume).cumsum()
    ad_ema_fast = ad_line.ewm(span=5, adjust=False).mean()
    ad_ema_slow = ad_line.ewm(span=20, adjust=False).mean()
    ad_osc = ad_ema_fast - ad_ema_slow
    ad_range = ad_osc.abs().rolling(40).max().replace(0, float("nan"))
    ad_signal = (ad_osc / ad_range).clip(-1, 1)

    obv_dir = np.sign(close.diff()).fillna(0)
    obv_raw = (obv_dir * volume).cumsum()
    obv_ema10 = obv_raw.ewm(span=10, adjust=False).mean()
    obv_ema30 = obv_raw.ewm(span=30, adjust=False).mean()
    obv_diff = obv_ema10 - obv_ema30
    obv_rng = obv_diff.abs().rolling(40).max().replace(0, float("nan"))
    obv_signal = (obv_diff / obv_rng).clip(-1, 1)

    vol_score = 0.35 * pv_align + 0.35 * ad_signal + 0.30 * obv_signal

    # LAYER 4: Mean-Reversion Extremes
    delta_c = close.diff()
    gain_c = delta_c.clip(lower=0).rolling(14).mean()
    loss_c = (-delta_c.clip(upper=0)).rolling(14).mean()
    rs_c = gain_c / loss_c.replace(0, float("nan"))
    rsi_raw = 100 - (100 / (1 + rs_c))

    rsi_extreme = np.where(rsi_raw < 25, (25 - rsi_raw) / 25,
                  np.where(rsi_raw > 75, (75 - rsi_raw) / 25, 0.0))
    rsi_extreme = pd.Series(rsi_extreme, index=close.index).clip(-1, 1)

    low14_c = low.rolling(14).min()
    high14_c = high.rolling(14).max()
    stoch_k = 100 * (close - low14_c) / (high14_c - low14_c).replace(0, float("nan"))
    stoch_k = stoch_k.rolling(3).mean()
    stoch_extreme = np.where(stoch_k < 20, (20 - stoch_k) / 20,
                    np.where(stoch_k > 80, (80 - stoch_k) / 20, 0.0))
    stoch_extreme = pd.Series(stoch_extreme, index=close.index).clip(-1, 1)

    sma20_c = close.rolling(20).mean()
    std20_c = close.rolling(20).std()
    bb_z = (close - sma20_c) / std20_c.replace(0, float("nan"))
    bb_extreme = np.where(bb_z < -2, (-2 - bb_z.clip(lower=-4)) / 2,
                 np.where(bb_z > 2, (2 - bb_z.clip(upper=4)) / 2, 0.0))
    bb_extreme = pd.Series(bb_extreme, index=close.index).clip(-1, 1)

    tp_c = (high + low + close) / 3
    tp_sma_c = tp_c.rolling(20).mean()
    tp_mad_c = tp_c.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci_raw = (tp_c - tp_sma_c) / (0.015 * tp_mad_c.replace(0, float("nan")))
    cci_extreme = np.where(cci_raw < -150, (cci_raw + 150).clip(-200, 0) / 200,
                  np.where(cci_raw > 150, (cci_raw - 150).clip(0, 200) / -200, 0.0))
    cci_extreme = pd.Series(cci_extreme, index=close.index).clip(-1, 1)

    mr_score = 0.35 * rsi_extreme + 0.30 * stoch_extreme + 0.20 * bb_extreme + 0.15 * cci_extreme

    # LAYER 5: Divergence
    lookback_div = 20
    price_slope = (close - close.shift(lookback_div)) / close.shift(lookback_div).replace(0, float("nan"))
    rsi_slope = (rsi_raw - rsi_raw.shift(lookback_div)) / 100
    obv_slope = (obv_raw - obv_raw.shift(lookback_div))
    obv_slope_n = obv_slope / obv_slope.abs().rolling(40).max().replace(0, float("nan"))
    rsi_div = np.sign(rsi_slope) - np.sign(price_slope)
    obv_div = np.sign(obv_slope_n) - np.sign(price_slope)
    div_raw = (rsi_div * 0.6 + obv_div * 0.4) / 4
    div_score = pd.Series(div_raw, index=close.index).clip(-0.5, 0.5)

    # LAYER 6: Trend Structure
    ema12_c = close.ewm(span=12, adjust=False).mean()
    ema26_c = close.ewm(span=26, adjust=False).mean()
    macd_line_c = ema12_c - ema26_c
    signal_c = macd_line_c.ewm(span=9, adjust=False).mean()
    hist_c = macd_line_c - signal_c
    hist_slope = hist_c.diff(3)
    hist_range = hist_c.abs().rolling(20).max().replace(0, float("nan"))
    macd_signal = (hist_c / hist_range).clip(-1, 1)
    macd_accel = (hist_slope / hist_range).clip(-1, 1)
    trend_macd = 0.6 * macd_signal + 0.4 * macd_accel

    tr_c = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    plus_dm_c = high.diff().clip(lower=0)
    minus_dm_c = (-low.diff()).clip(lower=0)
    plus_dm_c[plus_dm_c < minus_dm_c] = 0
    minus_dm_c[minus_dm_c < plus_dm_c] = 0
    atr14_c = tr_c.ewm(span=14, adjust=False).mean()
    plus_di_c = 100 * plus_dm_c.ewm(span=14, adjust=False).mean() / atr14_c.replace(0, float("nan"))
    minus_di_c = 100 * minus_dm_c.ewm(span=14, adjust=False).mean() / atr14_c.replace(0, float("nan"))
    dx_c = 100 * (plus_di_c - minus_di_c).abs() / (plus_di_c + minus_di_c).replace(0, float("nan"))
    adx_c = dx_c.ewm(span=14, adjust=False).mean()
    di_diff = (plus_di_c - minus_di_c) / (plus_di_c + minus_di_c).replace(0, float("nan"))
    adx_regime = ((adx_c - 20) / 30).clip(0, 1)
    trend_adx = di_diff * adx_regime
    trend_score = 0.55 * trend_macd + 0.45 * trend_adx

    # MASTER BLEND
    trend_w = adx_regime.fillna(0.4)
    mr_w = 1.0 - trend_w
    trend_blend = 0.40 * trend_score + 0.35 * mom_score + 0.15 * vol_score + 0.10 * div_score
    mr_blend = 0.40 * mr_score + 0.25 * div_score + 0.20 * vol_score + 0.15 * mom_score
    raw_signal = trend_w * trend_blend + mr_w * mr_blend
    vol_confirm = (np.sign(raw_signal) * np.sign(vol_score)).clip(0, 1)
    raw_signal = raw_signal * (0.80 + 0.20 * vol_confirm)
    raw_signal = raw_signal * vol_dampener
    raw_signal = raw_signal + 0.08 * vol_compression * np.sign(mom_score)
    raw_csi = (raw_signal * 100).clip(-100, 100)

    # Adaptive smoothing
    ema_fast = raw_csi.ewm(span=3, adjust=False).mean()
    ema_slow = raw_csi.ewm(span=8, adjust=False).mean()
    smooth_w = adx_regime.fillna(0.5)
    csi = smooth_w * ema_fast + (1 - smooth_w) * ema_slow

    return csi


def evaluate_csi(symbol: str, csi: pd.Series, close: pd.Series) -> dict:
    """Compute performance metrics for CSI on a single asset."""
    # Align and drop NaN
    valid = csi.notna() & close.notna()
    csi = csi[valid]
    close = close[valid]

    if len(csi) < 60:
        return {"symbol": symbol, "error": "insufficient data"}

    # Forward returns
    fwd_5 = close.pct_change(5).shift(-5)
    fwd_10 = close.pct_change(10).shift(-10)
    fwd_1 = close.pct_change(1).shift(-1)

    # Drop last few bars where we don't have forward returns
    mask = fwd_10.notna() & csi.notna()
    csi_v = csi[mask]
    fwd_5_v = fwd_5[mask]
    fwd_10_v = fwd_10[mask]
    fwd_1_v = fwd_1[mask]

    if len(csi_v) < 30:
        return {"symbol": symbol, "error": "insufficient valid data"}

    # 1. Correlation with forward returns
    corr_5d = csi_v.corr(fwd_5_v)
    corr_10d = csi_v.corr(fwd_10_v)

    # 2. Directional accuracy (does sign of CSI predict sign of forward return?)
    dir_correct_5d = ((np.sign(csi_v) == np.sign(fwd_5_v)) | (csi_v == 0)).mean()
    dir_correct_10d = ((np.sign(csi_v) == np.sign(fwd_10_v)) | (csi_v == 0)).mean()

    # 3. Buy signal metrics (CSI > 30)
    buy_mask = csi_v > 30
    sell_mask = csi_v < -30
    strong_buy = csi_v > 50
    strong_sell = csi_v < -50

    buy_count = buy_mask.sum()
    sell_count = sell_mask.sum()
    strong_buy_count = strong_buy.sum()
    strong_sell_count = strong_sell.sum()

    # Average forward return when signal fires
    buy_avg_5d = fwd_5_v[buy_mask].mean() * 100 if buy_count > 5 else np.nan
    sell_avg_5d = fwd_5_v[sell_mask].mean() * 100 if sell_count > 5 else np.nan
    strong_buy_avg = fwd_5_v[strong_buy].mean() * 100 if strong_buy_count > 3 else np.nan
    strong_sell_avg = fwd_5_v[strong_sell].mean() * 100 if strong_sell_count > 3 else np.nan

    # Hit rates (buy signal -> positive return)
    buy_hit = (fwd_5_v[buy_mask] > 0).mean() if buy_count > 5 else np.nan
    sell_hit = (fwd_5_v[sell_mask] < 0).mean() if sell_count > 5 else np.nan

    # 4. Simple signal-following strategy
    # Go long when CSI > 0, flat when CSI <= 0
    position = np.sign(csi_v).clip(0, 1)  # long-only
    strat_ret = position * fwd_1_v
    strat_ret = strat_ret.dropna()
    buy_hold_ret = fwd_1_v.dropna()

    # Sharpe (annualized)
    if len(strat_ret) > 20 and strat_ret.std() > 0:
        sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
    else:
        sharpe = np.nan

    bh_sharpe = (buy_hold_ret.mean() / buy_hold_ret.std()) * np.sqrt(252) if buy_hold_ret.std() > 0 else np.nan

    # Max drawdown of strategy
    cumret = (1 + strat_ret).cumprod()
    peak = cumret.cummax()
    dd = (cumret - peak) / peak
    max_dd = dd.min() * 100

    # CAGR
    if len(strat_ret) > 0:
        total_ret = cumret.iloc[-1] / cumret.iloc[0] if cumret.iloc[0] > 0 else 1
        years = len(strat_ret) / 252
        cagr = (total_ret ** (1 / max(years, 0.1)) - 1) * 100
    else:
        cagr = np.nan

    # Signal frequency
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
    print("=" * 90)
    print("  CSI v2 COMPREHENSIVE TEST HARNESS")
    print("  Universe: 50 stocks + GLD + SLV + BTC-USD")
    print("=" * 90)
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

        csi = compute_csi(df_reset)
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

    # Print per-asset results
    print(f"{'Symbol':<10} {'Corr5d':>7} {'Corr10d':>8} {'DirAcc5':>8} {'BuyHit':>7} {'SellHit':>8} "
          f"{'Sharpe':>7} {'B&H Sh':>7} {'CAGR':>6} {'MaxDD':>6} {'SigFreq':>8}")
    print("-" * 90)

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

    # Aggregate statistics
    print()
    print("=" * 90)
    print("  AGGREGATE STATISTICS")
    print("=" * 90)

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

    # Signal quality benchmarks
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

    # Strong signal analysis
    print()
    print("  STRONG SIGNAL ANALYSIS (|CSI| > 50):")
    sb_avgs = [r["strong_buy_avg"] for r in results if r["strong_buy_avg"] is not None]
    ss_avgs = [r["strong_sell_avg"] for r in results if r["strong_sell_avg"] is not None]
    print(f"    Strong Buy avg 5d ret: {np.mean(sb_avgs):.3f}% ({len(sb_avgs)} assets)" if sb_avgs else "    Strong Buy: insufficient data")
    print(f"    Strong Sell avg 5d ret:{np.mean(ss_avgs):.3f}% ({len(ss_avgs)} assets)" if ss_avgs else "    Strong Sell: insufficient data")

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
