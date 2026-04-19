"""
CSI v10: FINAL — Minimal Corrections for Maximum Sharpe
=========================================================
After testing v2-v9b, the winning formula is:
1. v3 core (proven buy/sell architecture)
2. Conditional oversold flip ONLY (the ONE correction that helps)
3. Moderate bullish bias (+6)
4. NO other corrections (they all hurt more than they help)

Key lessons from 8 iterations:
- Sell signals in a bull market CANNOT have >50% hit rate (structural)
- The oversold flip correctly reclassifies dip-buying opportunities
- Bullish bias captures more drift → higher Sharpe
- Simpler is better: every extra correction adds noise

Also testing v10b: v9b simple core (94% buy profitable) + bias + conditional flip
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
    """Exact v3 CSI core computation."""
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
    vol_20 = ret_1.rolling(20).std().replace(0, float("nan"))
    mom_fast = (close.pct_change(5) / (vol_20 * np.sqrt(5))).clip(-3, 3) / 3
    mom_med = (close.pct_change(10) / (vol_20 * np.sqrt(10))).clip(-3, 3) / 3
    mom_slow = (close.pct_change(20) / (vol_20 * np.sqrt(20))).clip(-3, 3) / 3
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

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    hist_range = hist.abs().rolling(20).max().replace(0, float("nan"))
    macd_n = (hist / hist_range).clip(-1, 1)
    hist_accel = hist.diff(3)
    macd_accel = (hist_accel / hist_range).clip(-1, 1)
    trend_macd = 0.6 * macd_n + 0.4 * macd_accel

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    atr14 = tr.ewm(span=14, adjust=False).mean()
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

    return {"csi": v3_csi, "rsi": rsi, "stoch_k": stoch_k, "above_200": above_200}


def compute_v10(df):
    """v3 + conditional oversold flip + bias +6. MINIMAL corrections."""
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

    # ONE correction: conditional oversold flip
    oversold = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    stoch_os = pd.Series(np.where(stoch_k < 20, (20 - stoch_k) / 20, 0.0), index=close.index).clip(0, 1)
    os_strength = np.maximum(oversold, stoch_os)
    csi = csi + os_strength * above_200 * 35
    csi = csi - os_strength * (1 - above_200) * 8

    # Bias +6
    csi = csi + 6

    return csi.clip(-100, 100)


def compute_v10b(df):
    """v9b simple core + bias +8 + conditional oversold flip.
    4 orthogonal signals, simple and robust."""
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
    above_200 = (close > sma_200).astype(float).fillna(0)

    # Signal 1: Trend (slope of SMA50, more robust than distance)
    ma50_slope = sma_50.pct_change(10)
    trend_sig = (ma50_slope / ma50_slope.abs().rolling(60, min_periods=20).max().replace(0, float("nan"))).clip(-1, 1)

    # Signal 2: Multi-TF Momentum (5d, 10d, 20d z-scores)
    mom5 = (close.pct_change(5) / (vol_20 * np.sqrt(5))).clip(-3, 3) / 3
    mom10 = (close.pct_change(10) / (vol_20 * np.sqrt(10))).clip(-3, 3) / 3
    mom20 = (close.pct_change(20) / (vol_20 * np.sqrt(20))).clip(-3, 3) / 3
    mom_sig = 0.45 * mom5 + 0.35 * mom10 + 0.20 * mom20

    # Signal 3: Volume Flow (up vol / down vol)
    up_vol = volume.where(close > close.shift(1), 0.0).rolling(10).sum()
    dn_vol = volume.where(close <= close.shift(1), 0.0).rolling(10).sum()
    vol_flow = ((up_vol - dn_vol) / (up_vol + dn_vol).replace(0, float("nan"))).clip(-1, 1)

    # Signal 4: RSI mean reversion (inverted, oversold = positive)
    delta_c = close.diff()
    gain = delta_c.clip(lower=0).rolling(14).mean()
    loss = (-delta_c.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    mr_sig = -(rsi - 50) / 50

    # Composite: weighted
    raw = (0.35 * trend_sig + 0.30 * mom_sig + 0.20 * vol_flow + 0.15 * mr_sig) * 100

    # Conditional oversold flip
    oversold = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    raw = raw + oversold * above_200 * 35
    raw = raw - oversold * (1 - above_200) * 8

    # Bias +8
    raw = raw + 8

    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def compute_v10c(df):
    """HYBRID: v3 core for signal quality, v9b ideas for simplicity.
    v3 core + conditional oversold flip + DYNAMIC bias (higher in uptrend)."""
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

    # Conditional oversold flip
    oversold = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    stoch_os = pd.Series(np.where(stoch_k < 20, (20 - stoch_k) / 20, 0.0), index=close.index).clip(0, 1)
    os_strength = np.maximum(oversold, stoch_os)
    csi = csi + os_strength * above_200 * 35
    csi = csi - os_strength * (1 - above_200) * 8

    # DYNAMIC bias: +8 in uptrend, +2 in downtrend
    # This keeps us in the market more during bull regimes
    # but doesn't blindly force positive CSI in bear regimes
    dynamic_bias = above_200 * 8 + (1 - above_200) * 2
    csi = csi + dynamic_bias

    return csi.clip(-100, 100)


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


def summarize(name, results):
    def sm(v):
        c = [x for x in v if x is not None and not np.isnan(x)]
        return np.mean(c) if c else float("nan")
    def smed(v):
        c = [x for x in v if x is not None and not np.isnan(x)]
        return np.median(c) if c else float("nan")

    n = len(results)
    s = [r["sharpe"] for r in results]
    bhs = [r["bh_sh"] for r in results]
    bh_r = [r["buy_hit"] for r in results]
    sh_r = [r["sell_hit"] for r in results]
    ba = [r["buy_avg"] for r in results]
    sa = [r["sell_avg"] for r in results]
    pp = [r["pct_pos"] for r in results]
    pr = [r["pos_ret"] for r in results]
    nr = [r["neg_ret"] for r in results]

    sb = sum(1 for r in results if r["sharpe"] is not None and r["bh_sh"] is not None and r["sharpe"] > r["bh_sh"])
    se = sum(1 for r in results if r["sharpe"] is not None and r["bh_sh"] is not None)
    bp = sum(1 for r in results if r["buy_avg"] is not None and r["buy_avg"] > 0)
    be = sum(1 for r in results if r["buy_avg"] is not None)
    sc = sum(1 for r in results if r["sell_avg"] is not None and r["sell_avg"] < 0)
    sce = sum(1 for r in results if r["sell_avg"] is not None)
    gs = sum(1 for r in results if r["pos_ret"] is not None and r["neg_ret"] is not None and r["pos_ret"] > r["neg_ret"])
    gse = sum(1 for r in results if r["pos_ret"] is not None and r["neg_ret"] is not None)
    spread = sm(pr) - sm(nr)

    print(f"\n  {name} ({n} assets)")
    print(f"    Median Sharpe:    {smed(s):.3f}  (B&H: {smed(bhs):.3f})")
    print(f"    Sharpe>B&H:       {sb}/{se} ({100*sb/max(se,1):.0f}%)")
    print(f"    Buy Hit:          {sm(bh_r):.1f}%  |  Buy Prof: {bp}/{be} ({100*bp/max(be,1):.0f}%)")
    print(f"    Sell Hit:         {sm(sh_r):.1f}%  |  Sell Corr: {sc}/{sce} ({100*sc/max(sce,1):.0f}%)")
    print(f"    CSI>0: {sm(pp):.0f}%  |  PosRet: {sm(pr):.1f}%  |  NegRet: {sm(nr):.1f}%  |  Spread: {spread:+.1f}%")
    print(f"    Good sep:         {gs}/{gse} ({100*gs/max(gse,1):.0f}%)")

    return {
        "name": name, "med_sh": smed(s), "sh_beat": f"{sb}/{se}",
        "buy_hit": sm(bh_r), "sell_hit": sm(sh_r),
        "buy_prof": f"{bp}/{be}", "sell_corr": f"{sc}/{sce}",
        "spread": spread, "pct_pos": sm(pp), "good_sep": f"{gs}/{gse}",
    }


def main():
    print("=" * 100)
    print("  CSI v10 — FINAL SHOOTOUT: v10 vs v10b vs v10c")
    print("=" * 100)

    configs = [
        ("v10:  v3+flip+bias6", compute_v10),
        ("v10b: Simple4+flip+bias8", compute_v10b),
        ("v10c: v3+flip+dynBias", compute_v10c),
    ]

    all_s = []
    for name, fn in configs:
        print(f"\n  Running {name}...")
        r = run_version(name, fn)
        s = summarize(name, r)
        all_s.append(s)

    print("\n\n" + "=" * 105)
    print("  FINAL COMPARISON")
    print("=" * 105)
    print(f"\n  {'Metric':<22} {'v3':>8} {'v7':>8} {'v8':>8} ", end="")
    for s in all_s:
        print(f" {s['name'][:12]:>12}", end="")
    print()
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8} ", end="")
    for _ in all_s:
        print(f" {'─'*12}", end="")
    print()

    refs = {"Med Sharpe": [0.487, 0.585, 0.527], "Sharpe>B&H": ["14/50", "14/50", "10/50"],
            "Buy Prof": ["41/50", "39/50", "40/50"], "Sell Corr": ["11/50", "10/32", "11/50"],
            "Buy Hit": [56.7, 56.3, 56.5], "Sell Hit": [41.0, 38.9, 42.6],
            "Regime Spr": ["N/A", -5.2, 3.2], "% in mkt": [55, 69, 54]}

    for metric, ref_vals in refs.items():
        print(f"  {metric:<22}", end="")
        for rv in ref_vals:
            if isinstance(rv, float):
                print(f" {rv:>8.1f}", end="")
            elif isinstance(rv, int):
                print(f" {rv:>8d}", end="")
            else:
                print(f" {str(rv):>8}", end="")
        for s in all_s:
            if metric == "Med Sharpe":
                print(f" {s['med_sh']:>12.3f}", end="")
            elif metric == "Sharpe>B&H":
                print(f" {s['sh_beat']:>12}", end="")
            elif metric == "Buy Prof":
                print(f" {s['buy_prof']:>12}", end="")
            elif metric == "Sell Corr":
                print(f" {s['sell_corr']:>12}", end="")
            elif metric == "Buy Hit":
                print(f" {s['buy_hit']:>11.1f}%", end="")
            elif metric == "Sell Hit":
                print(f" {s['sell_hit']:>11.1f}%", end="")
            elif metric == "Regime Spr":
                print(f" {s['spread']:>+11.1f}%", end="")
            elif metric == "% in mkt":
                print(f" {s['pct_pos']:>11.0f}%", end="")
        print()

    # Find best version
    best = max(all_s, key=lambda x: x["med_sh"])
    print(f"\n  >>> WINNER by Median Sharpe: {best['name']}")
    print()


if __name__ == "__main__":
    main()
