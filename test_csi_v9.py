"""
CSI v9: Best of v7 + v8 — Conditional Corrections + Modest Bullish Bias
=========================================================================
v7: Median Sharpe 0.585 (best), but regime spread -5.2% (wrong direction)
v8: Regime spread +3.2% (correct!), but Sharpe only 0.527

v9 goal: Conditional corrections from v8 + modest bias to increase market time
This should give positive regime spread AND high Sharpe.

Also testing: a RADICALLY SIMPLER approach alongside v8 corrections.
Instead of 7-section v3 core, what if we use fewer but more orthogonal signals?
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


def compute_v3_core(close, high, low, volume):
    """Exact v3 CSI computation. Returns (v3_csi, rsi, stoch_k, sma_200, above_200, adx_regime, ma50_slope, above_50)"""
    n = len(close)
    if n < 60:
        return None

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

    return {
        "csi": v3_csi, "rsi": rsi, "stoch_k": stoch_k, "sma_200": sma_200,
        "above_200": above_200, "above_50": above_50, "adx_regime": adx_regime,
        "ma50_slope": sma_50.pct_change(10), "sma_50": sma_50, "vol_flow": vol_flow,
        "mom_score": mom_score, "trend_score": trend_score,
    }


def compute_csi_v9(df):
    """v8 conditional corrections + modest bullish bias (+5) for more market time."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    core = compute_v3_core(close, high, low, volume)
    if core is None:
        return pd.Series(dtype=float)

    csi = core["csi"].copy()
    rsi = core["rsi"]
    stoch_k = core["stoch_k"]
    above_200 = core["above_200"].fillna(0)
    ma50_slope = core["ma50_slope"]

    # CORRECTION 1: Conditional oversold flip (from v8)
    oversold = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    stoch_os = pd.Series(np.where(stoch_k < 20, (20 - stoch_k) / 20, 0.0), index=close.index).clip(0, 1)
    os_strength = np.maximum(oversold, stoch_os)

    uptrend = above_200
    csi = csi + os_strength * uptrend * 40  # Boost in uptrend
    csi = csi - os_strength * (1 - uptrend) * 10  # Strengthen in downtrend

    # CORRECTION 2: Overbought with rollover (from v8, slightly reduced)
    rsi_ob = pd.Series(np.where(rsi > 68, (rsi - 68) / 32, 0.0), index=close.index).clip(0, 1)
    ma50_rolling_over = (ma50_slope < 0).astype(float)
    csi = csi - rsi_ob * ma50_rolling_over * 20

    # CORRECTION 3: RSI declining from overbought (lighter)
    rsi_max10 = rsi.rolling(10).max()
    rsi_was_ob = (rsi_max10 > 70).astype(float)
    rsi_drop = ((rsi_max10 - rsi) / 20).clip(0, 1)
    csi = csi - rsi_was_ob * rsi_drop * 10

    # CORRECTION 4: Bullish bias (modest +5)
    csi = csi + 5

    csi = csi.clip(-100, 100)
    csi = csi.ewm(span=2, adjust=False).mean()
    return csi


def compute_csi_v9b(df):
    """Radical simplification: 4 orthogonal z-score signals + conditional corrections.
    Testing if SIMPLER is BETTER."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    n = len(close)
    if n < 60:
        return pd.Series(dtype=float)

    ret_1 = close.pct_change(1)
    vol_20 = ret_1.rolling(20).std().replace(0, float("nan"))
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200, min_periods=60).mean()

    # SIGNAL 1: TREND (normalized distance from SMA50) [-1, +1]
    trend_raw = (close - sma_50) / (vol_20 * close).replace(0, float("nan"))
    trend_sig = trend_raw.rolling(60, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / max(x.std(), 1e-8), raw=False
    ).clip(-2, 2) / 2

    # SIGNAL 2: MOMENTUM (20-day risk-adjusted return) [-1, +1]
    ret_20 = close.pct_change(20)
    mom_sig = (ret_20 / (vol_20 * np.sqrt(20))).clip(-2, 2) / 2

    # SIGNAL 3: VOLUME FLOW (up volume vs down volume) [-1, +1]
    up_vol = volume.where(close > close.shift(1), 0.0).rolling(10).sum()
    dn_vol = volume.where(close <= close.shift(1), 0.0).rolling(10).sum()
    vol_flow = ((up_vol - dn_vol) / (up_vol + dn_vol).replace(0, float("nan"))).clip(-1, 1)

    # SIGNAL 4: MEAN REVERSION (RSI-based, inverted at extremes) [-1, +1]
    delta_c = close.diff()
    gain = delta_c.clip(lower=0).rolling(14).mean()
    loss = (-delta_c.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    # Inverted: oversold → positive, overbought → negative (mean reversion)
    mr_signal = -(rsi - 50) / 50  # -1 (overbought) to +1 (oversold)

    # COMPOSITE: Equal weight
    raw = 0.35 * trend_sig + 0.30 * mom_sig + 0.20 * vol_flow + 0.15 * mr_signal

    # Scale to [-100, 100]
    raw = raw * 100

    # CONDITIONAL CORRECTIONS
    above_200 = (close > sma_200).astype(float).fillna(0)

    # Oversold in uptrend: boost positive
    oversold = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    raw = raw + oversold * above_200 * 30
    raw = raw - oversold * (1 - above_200) * 8

    # Bullish bias
    raw = raw + 5

    raw = raw.clip(-100, 100)
    raw = raw.ewm(span=5, adjust=False).mean()
    return raw


def evaluate(symbol, csi, close):
    valid = csi.notna() & close.notna()
    csi = csi[valid]; close = close[valid]
    if len(csi) < 60:
        return None

    fwd_5 = close.pct_change(5).shift(-5)
    fwd_1 = close.pct_change(1).shift(-1)
    fwd_10 = close.pct_change(10).shift(-10)
    mask = fwd_10.notna() & csi.notna()
    cv = csi[mask]; f5 = fwd_5[mask]; f1 = fwd_1[mask]
    if len(cv) < 30:
        return None

    buy_m = cv > 30; sell_m = cv < -30
    bc = buy_m.sum(); sc = sell_m.sum()

    buy_avg = f5[buy_m].mean() * 100 if bc > 5 else np.nan
    sell_avg = f5[sell_m].mean() * 100 if sc > 5 else np.nan
    buy_hit = (f5[buy_m] > 0).mean() * 100 if bc > 5 else np.nan
    sell_hit = (f5[sell_m] < 0).mean() * 100 if sc > 5 else np.nan

    pos = np.sign(cv).clip(0, 1)
    sr = (pos * f1).dropna()
    bhr = f1.dropna()
    sharpe = (sr.mean() / sr.std()) * np.sqrt(252) if len(sr) > 20 and sr.std() > 0 else np.nan
    bh_sh = (bhr.mean() / bhr.std()) * np.sqrt(252) if bhr.std() > 0 else np.nan

    pct_p = (cv > 0).mean() * 100
    pr = f1[cv > 0].mean() * 252 * 100 if (cv > 0).sum() > 20 else np.nan
    nr = f1[cv <= 0].mean() * 252 * 100 if (cv <= 0).sum() > 20 else np.nan

    return {
        "sym": symbol, "buy_hit": buy_hit, "sell_hit": sell_hit,
        "buy_avg": buy_avg, "sell_avg": sell_avg,
        "sharpe": sharpe, "bh_sh": bh_sh,
        "pct_pos": pct_p, "pos_ret": pr, "neg_ret": nr,
        "bc": bc, "sc": sc,
    }


def run_version(name, compute_fn):
    results = []
    for sym in TEST_UNIVERSE:
        df = load_ohlcv(sym)
        if df.empty or len(df) < 100:
            continue
        df_r = df.reset_index(drop=True)
        for col in ["close", "high", "low", "volume"]:
            df_r[col] = df[col].astype(float).reset_index(drop=True)
        csi = compute_fn(df_r)
        if csi is None or (hasattr(csi, 'empty') and csi.empty):
            continue
        m = evaluate(sym, csi, df_r["close"])
        if m is not None:
            results.append(m)
    return results


def print_summary(name, results):
    def sm(vals):
        c = [v for v in vals if v is not None and not np.isnan(v)]
        return np.mean(c) if c else float("nan")
    def smed(vals):
        c = [v for v in vals if v is not None and not np.isnan(v)]
        return np.median(c) if c else float("nan")

    n = len(results)
    sharpes = [r["sharpe"] for r in results]
    bh_shs = [r["bh_sh"] for r in results]
    buy_hits = [r["buy_hit"] for r in results]
    sell_hits = [r["sell_hit"] for r in results]
    buy_avgs = [r["buy_avg"] for r in results]
    sell_avgs = [r["sell_avg"] for r in results]
    pos_rets = [r["pos_ret"] for r in results]
    neg_rets = [r["neg_ret"] for r in results]
    pct_pos = [r["pct_pos"] for r in results]

    sharpe_beat = sum(1 for r in results if r["sharpe"] is not None and r["bh_sh"] is not None and r["sharpe"] > r["bh_sh"])
    sharpe_elig = sum(1 for r in results if r["sharpe"] is not None and r["bh_sh"] is not None)
    buy_prof = sum(1 for r in results if r["buy_avg"] is not None and r["buy_avg"] > 0)
    buy_elig = sum(1 for r in results if r["buy_avg"] is not None)
    sell_corr = sum(1 for r in results if r["sell_avg"] is not None and r["sell_avg"] < 0)
    sell_elig = sum(1 for r in results if r["sell_avg"] is not None)
    good_sep = sum(1 for r in results if r["pos_ret"] is not None and r["neg_ret"] is not None and r["pos_ret"] > r["neg_ret"])
    sep_elig = sum(1 for r in results if r["pos_ret"] is not None and r["neg_ret"] is not None)

    spread = sm(pos_rets) - sm(neg_rets)

    print(f"\n  {name} — {n} assets")
    print(f"  {'─'*55}")
    print(f"    Median Sharpe:        {smed(sharpes):.3f}  (B&H: {smed(bh_shs):.3f})")
    print(f"    Mean Sharpe:          {sm(sharpes):.3f}")
    print(f"    Sharpe beats B&H:     {sharpe_beat}/{sharpe_elig} ({100*sharpe_beat/max(sharpe_elig,1):.0f}%)")
    print(f"    Mean Buy Hit:         {sm(buy_hits):.1f}%")
    print(f"    Mean Sell Hit:        {sm(sell_hits):.1f}%")
    print(f"    Buy profitable:       {buy_prof}/{buy_elig} ({100*buy_prof/max(buy_elig,1):.0f}%)")
    print(f"    Sell correct:         {sell_corr}/{sell_elig} ({100*sell_corr/max(sell_elig,1):.0f}%)")
    print(f"    Avg Buy 5d Ret:       {sm(buy_avgs):.3f}%")
    print(f"    Avg Sell 5d Ret:      {sm(sell_avgs):.3f}%")
    print(f"    CSI > 0:              {sm(pct_pos):.0f}% of time")
    print(f"    Ann ret (CSI>0):      {sm(pos_rets):.1f}%")
    print(f"    Ann ret (CSI<=0):     {sm(neg_rets):.1f}%")
    print(f"    Regime spread:        {spread:+.1f}%")
    print(f"    Good separation:      {good_sep}/{sep_elig} ({100*good_sep/max(sep_elig,1):.0f}%)")

    return {
        "name": name, "med_sharpe": smed(sharpes), "mean_sharpe": sm(sharpes),
        "sharpe_beat": f"{sharpe_beat}/{sharpe_elig}",
        "buy_hit": sm(buy_hits), "sell_hit": sm(sell_hits),
        "buy_prof": f"{buy_prof}/{buy_elig}", "sell_corr": f"{sell_corr}/{sell_elig}",
        "spread": spread, "pct_pos": sm(pct_pos),
    }


def main():
    print("=" * 100)
    print("  CSI v9 — HEAD-TO-HEAD: v9 (v8+bias) vs v9b (Radical Simplification)")
    print("=" * 100)

    print("\n  Computing v9 (v8 conditional corrections + bias +5)...")
    r9 = run_version("v9", compute_csi_v9)
    s9 = print_summary("v9: v8 Conditional + Bias(+5)", r9)

    print("\n  Computing v9b (Radical simplification: 4 signals + equal weight)...")
    r9b = run_version("v9b", compute_csi_v9b)
    s9b = print_summary("v9b: Radical Simple (4 signals)", r9b)

    print("\n")
    print("=" * 100)
    print("  COMPARISON TABLE")
    print("=" * 100)

    print(f"\n  ┌─────────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
    print(f"  │ Metric                  │    v3    │    v7    │    v8    │  v9 NOW  │  v9b     │")
    print(f"  ├─────────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")
    print(f"  │ Median Sharpe           │  0.487   │  0.585   │  0.527   │ {s9['med_sharpe']:>6.3f}   │ {s9b['med_sharpe']:>6.3f}   │")
    print(f"  │ Sharpe beats B&H        │  14/50   │  14/50   │  10/50   │ {s9['sharpe_beat']:>6s}   │ {s9b['sharpe_beat']:>6s}   │")
    print(f"  │ Buy profitable          │  41/50   │  39/50   │  40/50   │ {s9['buy_prof']:>6s}   │ {s9b['buy_prof']:>6s}   │")
    print(f"  │ Sell correct            │  11/50   │  10/32   │  11/50   │ {s9['sell_corr']:>6s}   │ {s9b['sell_corr']:>6s}   │")
    print(f"  │ Mean Buy Hit            │  56.7%   │  56.3%   │  56.5%   │ {s9['buy_hit']:>5.1f}%   │ {s9b['buy_hit']:>5.1f}%   │")
    print(f"  │ Mean Sell Hit           │  41.0%   │  38.9%   │  42.6%   │ {s9['sell_hit']:>5.1f}%   │ {s9b['sell_hit']:>5.1f}%   │")
    print(f"  │ Regime spread           │   N/A    │  -5.2%   │  +3.2%   │ {s9['spread']:>+5.1f}%   │ {s9b['spread']:>+5.1f}%   │")
    print(f"  │ % time in market        │   ~55%   │   69%    │   54%    │ {s9['pct_pos']:>5.0f}%   │ {s9b['pct_pos']:>5.0f}%   │")
    print(f"  └─────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")
    print()


if __name__ == "__main__":
    main()
