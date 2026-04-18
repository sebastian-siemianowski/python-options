"""
CSI v12: FINAL fine-grained bias sweep + adaptive bias.
v8+0 is the clear winner with Sharpe*Spread score of 0.544.
Test bias 1, 2, and adaptive (above_200*4) for a possible sweet spot.
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


def compute_v8_full(close, high, low, volume, bias=0.0, adaptive_bias=False):
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

    # ═══ v8 CORRECTIONS ═══
    csi = v3_csi.copy()

    # C1: Conditional oversold flip
    oversold = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    stoch_os = pd.Series(np.where(stoch_k < 20, (20 - stoch_k) / 20, 0.0), index=close.index).clip(0, 1)
    os_strength = np.maximum(oversold, stoch_os)
    uptrend = above_200.fillna(0)
    csi = csi + os_strength * uptrend * 40
    csi = csi - os_strength * (1 - uptrend) * 10

    # C2: Overbought with rollover
    rsi_ob = pd.Series(np.where(rsi > 68, (rsi - 68) / 32, 0.0), index=close.index).clip(0, 1)
    ma50_rolling_over = (ma50_slope < 0).astype(float)
    csi = csi - rsi_ob * ma50_rolling_over * 25
    rsi_max10 = rsi.rolling(10).max()
    rsi_was_ob = (rsi_max10 > 70).astype(float)
    rsi_declining_fast = ((rsi_max10 - rsi) / 20).clip(0, 1)
    csi = csi - rsi_was_ob * rsi_declining_fast * 12

    # C3: Volume divergence
    price_near_high = ((close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min()).replace(0, float("nan")))
    price_near_high = price_near_high.clip(0, 1)
    vol_10 = volume.rolling(10).mean()
    vol_30 = volume.rolling(30).mean()
    vol_declining = ((vol_30 - vol_10) / vol_30.replace(0, float("nan"))).clip(0, 1)
    csi = csi - (price_near_high > 0.70).astype(float) * vol_declining * 8

    # C4: Breakdown accelerator
    below_50 = (1 - above_50).astype(float)
    vol_increasing = (vol_10 > vol_30 * 1.1).astype(float)
    neg_csi = (csi < 0).astype(float)
    csi = csi - below_50 * vol_increasing * neg_csi * 8

    # Apply bias
    if adaptive_bias:
        csi = csi + uptrend * 4
    else:
        csi = csi + bias

    csi = csi.clip(-100, 100)
    csi = csi.ewm(span=2, adjust=False).mean()
    return csi


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


def run_test(name, bias=0.0, adaptive=False):
    results = []
    for sym in TEST_UNIVERSE:
        df = load_ohlcv(sym)
        if df.empty or len(df) < 100:
            continue
        close = df["close"].astype(float).reset_index(drop=True)
        high = df["high"].astype(float).reset_index(drop=True)
        low_s = df["low"].astype(float).reset_index(drop=True)
        vol = df["volume"].astype(float).reset_index(drop=True)
        csi = compute_v8_full(close, high, low_s, vol, bias=bias, adaptive_bias=adaptive)
        if csi is None:
            continue
        m = evaluate(sym, csi, close)
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
    pp = [r["pct_pos"] for r in results]
    pr = [r["pos_ret"] for r in results]
    nr = [r["neg_ret"] for r in results]
    bh_r = [r["buy_hit"] for r in results]
    sh_r = [r["sell_hit"] for r in results]

    sb = sum(1 for r in results if r["sharpe"] is not None and r["bh_sh"] is not None and r["sharpe"] > r["bh_sh"])
    se = sum(1 for r in results if r["sharpe"] is not None and r["bh_sh"] is not None)
    bp = sum(1 for r in results if r["buy_avg"] is not None and r["buy_avg"] > 0)
    be = sum(1 for r in results if r["buy_avg"] is not None)
    sc = sum(1 for r in results if r["sell_avg"] is not None and r["sell_avg"] < 0)
    sce = sum(1 for r in results if r["sell_avg"] is not None)
    gs = sum(1 for r in results if r["pos_ret"] is not None and r["neg_ret"] is not None and r["pos_ret"] > r["neg_ret"])
    gse = sum(1 for r in results if r["pos_ret"] is not None and r["neg_ret"] is not None)
    spread = sm(pr) - sm(nr)
    msh = smed(s)
    combined = msh * (1 + spread / 100)

    print(f"  {name:<18} Sh={msh:.3f}  Spr={spread:+5.1f}%  Comb={combined:.3f}  "
          f"BuyP={bp}/{be}  SellC={sc}/{sce}  SH%={sm(sh_r):.1f}%  GS={gs}/{gse}  Mkt={sm(pp):.0f}%")
    return {"name": name, "med_sh": msh, "spread": spread, "combined": combined,
            "buy_prof": f"{bp}/{be}", "sell_corr": f"{sc}/{sce}", "sell_hit": sm(sh_r),
            "good_sep": f"{gs}/{gse}", "pct_pos": sm(pp)}


def main():
    print("=" * 100)
    print("  CSI v12 — Fine-grained bias sweep (0, 1, 2) + adaptive bias")
    print("  Reference: v8+0 has Combined=0.544 (Sharpe=0.527, Spread=+3.2%)")
    print("=" * 100)

    configs = [
        ("v8+bias0", 0, False),
        ("v8+bias1", 1, False),
        ("v8+bias2", 2, False),
        ("v8+adaptive(4)", 0, True),
    ]

    print(f"\n  {'Name':<18} {'Sharpe':>8} {'Spread':>8} {'Combined':>10}  {'BuyP':>8} {'SellC':>8} {'SellH':>8} {'GS':>8} {'Mkt':>6}")
    print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*10}  {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")

    all_s = []
    for name, bias, adaptive in configs:
        r = run_test(name, bias=bias, adaptive=adaptive)
        s = summarize(name, r)
        all_s.append(s)

    best = max(all_s, key=lambda x: x["combined"])
    print(f"\n  >>> WINNER: {best['name']} (Combined={best['combined']:.3f})")
    print()


if __name__ == "__main__":
    main()
