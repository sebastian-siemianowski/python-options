#!/usr/bin/env python3
"""
CSI MEGA HARNESS — Systematic Strategy Exploration v13-v100
============================================================
Master test framework for iterating CSI from v13 through v100.
Tests each strategy on 100+ assets with comprehensive metrics.

Runs strategies in batches, tracks leaderboard, combines winners.
"""

import os, sys, warnings, time, json
warnings.filterwarnings("ignore")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
from functools import lru_cache

# ═══════════════════════════════════════════════════════════
# EXPANDED UNIVERSE: 101 assets
# ═══════════════════════════════════════════════════════════
UNIVERSE = [
    # Original 53
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "CRM", "ADBE",
    "NFLX", "CRWD", "JPM", "BAC", "GS", "MS", "SCHW", "AFRM", "JNJ", "UNH",
    "PFE", "ABBV", "MRNA", "LMT", "RTX", "NOC", "GD", "CAT", "DE", "BA",
    "UPS", "GE", "XOM", "CVX", "COP", "SLB", "HD", "NKE", "SBUX", "PG",
    "KO", "COST", "UPST", "IONQ", "DKNG", "SNAP", "SPY", "QQQ", "IWM", "DIA",
    "GLD", "SLV", "BTC-USD",
    # Additional 48 popular stocks
    "AVGO", "ORCL", "INTC", "CSCO", "IBM", "QCOM", "TXN", "MU", "AMAT", "LRCX",
    "MRVL", "ON", "NET", "DDOG", "SNOW", "PLTR", "SHOP", "PYPL", "V", "MA",
    "AXP", "BRK-B", "WFC", "C", "BLK", "T", "VZ", "TMUS", "DIS", "CMCSA",
    "WMT", "TGT", "LOW", "MCD", "UBER", "COIN", "MSTR", "SOFI", "HOOD", "ARM",
    "LLY", "MRK", "BMY", "GILD", "AMGN", "TMO", "DHR", "ABT",
]
PRICES_DIR = os.path.join(SRC_DIR, "data", "prices")
LEADERBOARD_FILE = os.path.join(SCRIPT_DIR, "csi_leaderboard.json")


# ═══════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ═══════════════════════════════════════════════════════════
_data_cache = {}

def load_ohlcv(symbol):
    if symbol in _data_cache:
        return _data_cache[symbol]
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
            df = df.dropna(subset=["close", "high", "low", "volume"]).reset_index(drop=True)
            _data_cache[symbol] = df
            return df
    _data_cache[symbol] = pd.DataFrame()
    return pd.DataFrame()


def preload_all():
    """Preload all data once."""
    loaded = 0
    for sym in UNIVERSE:
        df = load_ohlcv(sym)
        if not df.empty and len(df) >= 100:
            loaded += 1
    return loaded


# ═══════════════════════════════════════════════════════════
# COMMON TECHNICAL INDICATORS (computed once per symbol)
# ═══════════════════════════════════════════════════════════
_indicator_cache = {}

def get_indicators(symbol):
    """Compute all base indicators once, cache them."""
    if symbol in _indicator_cache:
        return _indicator_cache[symbol]

    df = load_ohlcv(symbol)
    if df.empty or len(df) < 60:
        _indicator_cache[symbol] = None
        return None

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    n = len(close)

    ind = {"close": close, "high": high, "low": low, "volume": volume, "n": n}

    # Moving averages
    ind["sma_10"] = close.rolling(10).mean()
    ind["sma_20"] = close.rolling(20).mean()
    ind["sma_50"] = close.rolling(50).mean()
    ind["sma_200"] = close.rolling(200, min_periods=60).mean()
    ind["ema_10"] = close.ewm(span=10, adjust=False).mean()
    ind["ema_21"] = close.ewm(span=21, adjust=False).mean()
    ind["ema_50"] = close.ewm(span=50, adjust=False).mean()

    ind["above_20"] = (close > ind["sma_20"]).astype(float)
    ind["above_50"] = (close > ind["sma_50"]).astype(float)
    ind["above_200"] = (close > ind["sma_200"]).astype(float)

    # MA slopes
    ind["ma20_slope"] = ind["sma_20"].pct_change(5)
    ind["ma20_slope_n"] = (ind["ma20_slope"] / ind["ma20_slope"].abs().rolling(60).max().replace(0, float("nan"))).clip(-1, 1)
    ind["ma50_slope"] = ind["sma_50"].pct_change(10)
    ind["ma50_slope_n"] = (ind["ma50_slope"] / ind["ma50_slope"].abs().rolling(60).max().replace(0, float("nan"))).clip(-1, 1)

    # MA context
    ind["ma_context"] = (0.25 * (ind["above_20"] * 2 - 1) + 0.25 * (ind["above_50"] * 2 - 1) +
                         0.25 * ind["ma20_slope_n"] + 0.25 * ind["ma50_slope_n"])

    # Returns & volatility
    ind["ret_1"] = close.pct_change(1)
    ind["vol_20"] = ind["ret_1"].rolling(20).std().replace(0, float("nan"))
    ind["vol_60"] = ind["ret_1"].rolling(60).std().replace(0, float("nan"))

    # Multi-TF momentum (vol-normalized z-scores)
    ind["mom_5"] = (close.pct_change(5) / (ind["vol_20"] * np.sqrt(5))).clip(-3, 3) / 3
    ind["mom_10"] = (close.pct_change(10) / (ind["vol_20"] * np.sqrt(10))).clip(-3, 3) / 3
    ind["mom_20"] = (close.pct_change(20) / (ind["vol_20"] * np.sqrt(20))).clip(-3, 3) / 3
    ind["mom_40"] = (close.pct_change(40) / (ind["vol_20"] * np.sqrt(40))).clip(-3, 3) / 3

    raw_mom = 0.50 * ind["mom_5"] + 0.30 * ind["mom_10"] + 0.20 * ind["mom_20"]
    signs_mom = pd.concat([np.sign(ind["mom_5"]), np.sign(ind["mom_10"]), np.sign(ind["mom_20"])], axis=1)
    agreement = signs_mom.sum(axis=1).abs() / 3.0
    ind["mom_score"] = raw_mom * (0.5 + 0.5 * agreement)
    ind["mom_accel"] = ind["mom_5"] - ind["mom_5"].shift(5)
    ind["mom_accel_n"] = (ind["mom_accel"] / ind["mom_accel"].abs().rolling(30).max().replace(0, float("nan"))).clip(-1, 1)

    # Volume flow
    up_vol = volume.where(close > close.shift(1), 0.0).rolling(10).sum()
    dn_vol = volume.where(close <= close.shift(1), 0.0).rolling(10).sum()
    ind["vol_ratio"] = ((up_vol - dn_vol) / (up_vol + dn_vol).replace(0, float("nan"))).clip(-1, 1)

    obv_dir = np.sign(close.diff()).fillna(0)
    obv_raw = (obv_dir * volume).cumsum()
    obv_ema_f = obv_raw.ewm(span=10, adjust=False).mean()
    obv_ema_s = obv_raw.ewm(span=30, adjust=False).mean()
    obv_diff = obv_ema_f - obv_ema_s
    obv_range = obv_diff.abs().rolling(40).max().replace(0, float("nan"))
    ind["obv_signal"] = (obv_diff / obv_range).clip(-1, 1)

    clv = ((close - low) - (high - close)) / (high - low).replace(0, float("nan"))
    ad_line = (clv * volume).cumsum()
    ad_ema5 = ad_line.ewm(span=5, adjust=False).mean()
    ad_ema20 = ad_line.ewm(span=20, adjust=False).mean()
    ad_osc = ad_ema5 - ad_ema20
    ad_rng = ad_osc.abs().rolling(40).max().replace(0, float("nan"))
    ind["ad_score"] = (ad_osc / ad_rng).clip(-1, 1)
    ind["vol_flow"] = 0.40 * ind["vol_ratio"] + 0.30 * ind["obv_signal"] + 0.30 * ind["ad_score"]

    # RSI
    delta_c = close.diff()
    gain_c = delta_c.clip(lower=0).rolling(14).mean()
    loss_c = (-delta_c.clip(upper=0)).rolling(14).mean()
    rs_c = gain_c / loss_c.replace(0, float("nan"))
    ind["rsi"] = 100 - (100 / (1 + rs_c))

    # Stochastic
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    ind["stoch_k"] = (100 * (close - low14) / (high14 - low14).replace(0, float("nan"))).rolling(3).mean()

    # Oscillator buy/sell
    rsi_buy = np.where(ind["rsi"] < 35, (35 - ind["rsi"]) / 35, 0.0)
    stoch_buy = np.where(ind["stoch_k"] < 25, (25 - ind["stoch_k"]) / 25, 0.0)
    ind["osc_buy"] = pd.Series(0.55 * rsi_buy + 0.45 * stoch_buy, index=close.index).clip(0, 1)

    rsi_declining = (ind["rsi"] < ind["rsi"].shift(3)).astype(float)
    stoch_declining = (ind["stoch_k"] < ind["stoch_k"].shift(3)).astype(float)
    rsi_sell = np.where((ind["rsi"] > 70) & (rsi_declining > 0), (ind["rsi"] - 70) / 30, 0.0)
    stoch_sell = np.where((ind["stoch_k"] > 80) & (stoch_declining > 0), (ind["stoch_k"] - 80) / 20, 0.0)
    ind["osc_sell"] = pd.Series(0.55 * rsi_sell + 0.45 * stoch_sell, index=close.index).clip(0, 1)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    hist_range = hist.abs().rolling(20).max().replace(0, float("nan"))
    ind["macd_n"] = (hist / hist_range).clip(-1, 1)
    hist_accel = hist.diff(3)
    ind["macd_accel"] = (hist_accel / hist_range).clip(-1, 1)
    ind["trend_macd"] = 0.6 * ind["macd_n"] + 0.4 * ind["macd_accel"]
    ind["macd_line"] = macd_line
    ind["macd_signal"] = signal_line
    ind["macd_hist"] = hist

    # ADX
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    ind["atr14"] = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / ind["atr14"].replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / ind["atr14"].replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    ind["adx"] = dx.ewm(span=14, adjust=False).mean()
    di_diff = (plus_di - minus_di) / (plus_di + minus_di).replace(0, float("nan"))
    ind["adx_regime"] = ((ind["adx"] - 15) / 35).clip(0, 1)
    ind["trend_adx"] = di_diff * ind["adx_regime"]
    ind["trend_score"] = 0.55 * ind["trend_macd"] + 0.45 * ind["trend_adx"]

    # Vol percentile
    ind["vol_pct"] = ind["vol_20"].rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
    ind["vol_dampener"] = 1.0 - 0.3 * (ind["vol_pct"] - 0.5).clip(0, 0.5)

    # Bollinger Bands
    bb_sma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    ind["bb_upper"] = bb_sma + 2 * bb_std
    ind["bb_lower"] = bb_sma - 2 * bb_std
    ind["bb_pctb"] = ((close - ind["bb_lower"]) / (ind["bb_upper"] - ind["bb_lower"]).replace(0, float("nan"))).clip(-0.5, 1.5)
    ind["bb_width"] = (4 * bb_std / bb_sma).replace(0, float("nan"))
    ind["bb_squeeze"] = ind["bb_width"].rolling(120, min_periods=20).rank(pct=True).fillna(0.5)

    # Keltner Channel
    kc_mid = close.ewm(span=20, adjust=False).mean()
    kc_atr = tr.rolling(10).mean()
    ind["kc_upper"] = kc_mid + 1.5 * kc_atr
    ind["kc_lower"] = kc_mid - 1.5 * kc_atr
    ind["kc_pct"] = ((close - ind["kc_lower"]) / (ind["kc_upper"] - ind["kc_lower"]).replace(0, float("nan"))).clip(-0.5, 1.5)

    # Williams %R
    ind["willr"] = -100 * (high14 - close) / (high14 - low14).replace(0, float("nan"))

    # CCI
    tp = (high + low + close) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    ind["cci"] = ((tp - tp_sma) / (0.015 * tp_mad).replace(0, float("nan"))).clip(-300, 300)

    # MFI (Money Flow Index)
    tp2 = (high + low + close) / 3
    mf_raw = tp2 * volume
    mf_pos = mf_raw.where(tp2 > tp2.shift(1), 0.0).rolling(14).sum()
    mf_neg = mf_raw.where(tp2 <= tp2.shift(1), 0.0).rolling(14).sum()
    mf_ratio = mf_pos / mf_neg.replace(0, float("nan"))
    ind["mfi"] = 100 - (100 / (1 + mf_ratio))

    # Rate of Change
    ind["roc_10"] = close.pct_change(10) * 100
    ind["roc_20"] = close.pct_change(20) * 100

    # ATR percent
    ind["atr_pct"] = (ind["atr14"] / close * 100).replace(0, float("nan"))

    # Higher highs / lower lows (structure)
    ind["hh"] = (high > high.rolling(20).max().shift(1)).astype(float)
    ind["ll"] = (low < low.rolling(20).min().shift(1)).astype(float)
    ind["hl_ratio"] = ind["hh"].rolling(10).sum() - ind["ll"].rolling(10).sum()

    # Volume relative
    ind["vol_rel"] = (volume / volume.rolling(20).mean().replace(0, float("nan"))).clip(0, 5)

    # Price channels
    ind["donch_high"] = high.rolling(20).max()
    ind["donch_low"] = low.rolling(20).min()
    ind["donch_pct"] = ((close - ind["donch_low"]) / (ind["donch_high"] - ind["donch_low"]).replace(0, float("nan"))).clip(0, 1)

    # Ichimoku (simplified)
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    ind["ichi_tk"] = ((tenkan - kijun) / close * 100).clip(-5, 5)  # Tenkan-Kijun diff
    ind["ichi_above_cloud"] = (close > kijun).astype(float)

    # Hurst exponent estimate (simple, vectorized)
    def _hurst_simple(s, window=100):
        """Simplified R/S Hurst estimate using numpy arrays."""
        vals = s.values.astype(float)
        out = np.full(len(vals), np.nan)
        log_w = np.log(window)
        for i in range(window, len(vals)):
            seg = vals[i-window:i]
            if np.any(np.isnan(seg)):
                continue
            mean_seg = np.mean(seg)
            cumdev = np.cumsum(seg - mean_seg)
            r = np.max(cumdev) - np.min(cumdev)
            std_seg = np.std(seg)
            if std_seg > 0 and r > 0:
                out[i] = np.log(r / std_seg) / log_w
        return pd.Series(out, index=s.index)

    ind["hurst"] = _hurst_simple(ind["ret_1"], window=100)

    # Efficiency ratio (Kaufman)
    direction = (close - close.shift(10)).abs()
    volatility_sum = ind["ret_1"].abs().rolling(10).sum()
    ind["efficiency"] = (direction / volatility_sum.replace(0, float("nan"))).clip(0, 1)

    _indicator_cache[symbol] = ind
    return ind


# ═══════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════
def evaluate(symbol, csi, close):
    """Evaluate CSI quality. Returns dict of metrics or None."""
    if csi is None:
        return None
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

    # Strategy: long when CSI > 0, flat otherwise
    pos = np.sign(cv).clip(0, 1)
    sr = (pos * f1).dropna()
    bhr = f1.dropna()
    sharpe = (sr.mean() / sr.std()) * np.sqrt(252) if len(sr) > 20 and sr.std() > 0 else np.nan
    bh_sh = (bhr.mean() / bhr.std()) * np.sqrt(252) if bhr.std() > 0 else np.nan

    pct_p = (cv > 0).mean() * 100
    pr = f1[cv > 0].mean() * 252 * 100 if (cv > 0).sum() > 20 else np.nan
    nr = f1[cv <= 0].mean() * 252 * 100 if (cv <= 0).sum() > 20 else np.nan

    # Max drawdown of strategy
    cum = (1 + sr).cumprod()
    peak = cum.cummax()
    dd = ((cum - peak) / peak).min() * 100 if len(cum) > 0 else np.nan

    return {
        "sym": symbol, "buy_hit": buy_hit, "sell_hit": sell_hit,
        "buy_avg": buy_avg, "sell_avg": sell_avg,
        "sharpe": sharpe, "bh_sh": bh_sh,
        "pct_pos": pct_p, "pos_ret": pr, "neg_ret": nr,
        "max_dd": dd,
    }


def summarize(name, results, verbose=True):
    """Compute aggregate metrics from per-asset results."""
    def sm(v):
        c = [x for x in v if x is not None and not np.isnan(x)]
        return np.mean(c) if c else float("nan")
    def smed(v):
        c = [x for x in v if x is not None and not np.isnan(x)]
        return np.median(c) if c else float("nan")

    n = len(results)
    if n == 0:
        return {"name": name, "med_sh": float("nan"), "spread": float("nan"), "combined": float("nan")}

    s = [r["sharpe"] for r in results]
    bhs = [r["bh_sh"] for r in results]
    bh_r = [r["buy_hit"] for r in results]
    sh_r = [r["sell_hit"] for r in results]
    pp = [r["pct_pos"] for r in results]
    pr = [r["pos_ret"] for r in results]
    nr = [r["neg_ret"] for r in results]
    dd = [r["max_dd"] for r in results]

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
    combined = msh * (1 + spread / 100) if not np.isnan(msh) and not np.isnan(spread) else float("nan")

    out = {
        "name": name, "n": n, "med_sh": msh, "mean_sh": sm(s),
        "spread": spread, "combined": combined,
        "buy_hit": sm(bh_r), "sell_hit": sm(sh_r),
        "buy_prof": f"{bp}/{be}", "sell_corr": f"{sc}/{sce}",
        "sharpe_beat": f"{sb}/{se}", "good_sep": f"{gs}/{gse}",
        "pct_pos": sm(pp), "max_dd": smed(dd),
    }

    if verbose:
        gs_pct = 100 * gs / max(gse, 1)
        print(f"  {name:<30} Sh={msh:6.3f} Spr={spread:+5.1f}% C={combined:6.3f} "
              f"BuyP={bp}/{be} SellC={sc}/{sce} SH%={sm(sh_r):4.1f}% "
              f"GS={gs}/{gse}({gs_pct:.0f}%) Mkt={sm(pp):.0f}% DD={smed(dd):.1f}%")

    return out


def run_strategy(name, compute_fn, verbose=True):
    """Run a strategy across all assets and return summary."""
    results = []
    for sym in UNIVERSE:
        ind = get_indicators(sym)
        if ind is None:
            continue
        try:
            csi = compute_fn(ind)
        except Exception:
            continue
        if csi is None or (hasattr(csi, 'empty') and csi.empty):
            continue
        m = evaluate(sym, csi, ind["close"])
        if m is not None:
            results.append(m)
    return summarize(name, results, verbose=verbose)


# ═══════════════════════════════════════════════════════════
# V8+1 BASELINE (current champion)
# ═══════════════════════════════════════════════════════════
def v8_baseline(ind):
    """v8+bias1: Current champion. Combined=0.547"""
    close = ind["close"]

    buy_raw = (
        0.30 * ind["trend_score"].clip(0, None) + 0.25 * ind["mom_score"].clip(0, None) +
        0.15 * ind["osc_buy"] + 0.15 * ind["vol_flow"].clip(0, None) +
        0.10 * ind["ma_context"].clip(0, None) + 0.05 * ind["mom_accel_n"].clip(0, None)
    )
    sell_raw = (
        0.25 * (-ind["trend_score"]).clip(0, None) + 0.25 * (-ind["mom_score"]).clip(0, None) +
        0.15 * ind["osc_sell"] + 0.15 * (-ind["vol_flow"]).clip(0, None) +
        0.10 * (-ind["ma_context"]).clip(0, None) + 0.05 * (-ind["mom_accel_n"]).clip(0, None) +
        0.05 * (1 - ind["above_50"]) * (-ind["trend_score"]).clip(0, None)
    )
    raw_csi = (buy_raw - sell_raw) * 100
    vol_confirm = (np.sign(raw_csi) * np.sign(ind["vol_flow"])).clip(0, 1)
    raw_csi = raw_csi * (0.80 + 0.20 * vol_confirm) * ind["vol_dampener"]

    buy_factors = pd.concat([
        (ind["trend_score"] > 0.05).astype(float), (ind["mom_score"] > 0.05).astype(float),
        (ind["vol_flow"] > 0.05).astype(float), (ind["ma_context"] > 0).astype(float),
    ], axis=1).sum(axis=1)
    sell_factors = pd.concat([
        (ind["trend_score"] < -0.05).astype(float), (ind["mom_score"] < -0.05).astype(float),
        (ind["vol_flow"] < -0.05).astype(float), (ind["ma_context"] < 0).astype(float),
    ], axis=1).sum(axis=1)
    buy_gate = (buy_factors >= 2).astype(float)
    sell_gate = (sell_factors >= 2).astype(float)
    gate = pd.Series(np.where(raw_csi > 0, buy_gate, np.where(raw_csi < 0, sell_gate, 0.5)), index=close.index)
    raw_csi = raw_csi * (0.3 + 0.7 * gate)
    raw_csi = raw_csi.clip(-100, 100)

    ema_f = raw_csi.ewm(span=3, adjust=False).mean()
    ema_s = raw_csi.ewm(span=8, adjust=False).mean()
    sw = ind["adx_regime"].fillna(0.5)
    csi = sw * ema_f + (1 - sw) * ema_s

    # v8 corrections
    rsi = ind["rsi"]; stoch_k = ind["stoch_k"]; above_200 = ind["above_200"].fillna(0)
    oversold_c = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    stoch_os = pd.Series(np.where(stoch_k < 20, (20 - stoch_k) / 20, 0.0), index=close.index).clip(0, 1)
    os_strength = np.maximum(oversold_c, stoch_os)
    csi = csi + os_strength * above_200 * 40 - os_strength * (1 - above_200) * 10

    rsi_ob = pd.Series(np.where(rsi > 68, (rsi - 68) / 32, 0.0), index=close.index).clip(0, 1)
    csi = csi - rsi_ob * (ind["ma50_slope"] < 0).astype(float) * 25
    rsi_max10 = rsi.rolling(10).max()
    csi = csi - (rsi_max10 > 70).astype(float) * ((rsi_max10 - rsi) / 20).clip(0, 1) * 12

    price_near_high = ((close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min()).replace(0, float("nan"))).clip(0, 1)
    vol_10 = ind["volume"].rolling(10).mean()
    vol_30 = ind["volume"].rolling(30).mean()
    csi = csi - (price_near_high > 0.70).astype(float) * ((vol_30 - vol_10) / vol_30.replace(0, float("nan"))).clip(0, 1) * 8
    csi = csi - (1 - ind["above_50"]).astype(float) * (vol_10 > vol_30 * 1.1).astype(float) * (csi < 0).astype(float) * 8

    csi = (csi + 1).clip(-100, 100).ewm(span=2, adjust=False).mean()
    return csi


# ═══════════════════════════════════════════════════════════
# STRATEGY LIBRARY: v13-v100
# ═══════════════════════════════════════════════════════════

def v13_kalman_trend(ind):
    """v13: Kalman-style adaptive smoothing on price, extract trend slope as signal."""
    close = ind["close"]
    # Simple exponential state estimation (alpha adapts to efficiency)
    alpha = ind["efficiency"].clip(0.05, 0.5).fillna(0.1)
    state = close.copy()
    vals = close.values.copy().astype(float)
    a = alpha.values
    for i in range(1, len(vals)):
        if not np.isnan(vals[i]) and not np.isnan(a[i]):
            vals[i] = a[i] * close.values[i] + (1 - a[i]) * vals[i-1]
    state = pd.Series(vals, index=close.index)
    # Signal = slope of state (normalized)
    slope = state.pct_change(5)
    slope_n = (slope / slope.abs().rolling(60, min_periods=20).max().replace(0, float("nan"))).clip(-1, 1)
    # Combine with volume confirmation
    raw = (0.6 * slope_n + 0.25 * ind["vol_flow"] + 0.15 * ind["mom_accel_n"]) * 60
    raw = raw + ind["above_200"].fillna(0) * 5 - (1 - ind["above_200"].fillna(0)) * 3
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v14_mean_reversion(ind):
    """v14: Pure mean-reversion. Buy oversold, sell overbought, with trend filter."""
    close = ind["close"]; rsi = ind["rsi"]; bb = ind["bb_pctb"]
    above_200 = ind["above_200"].fillna(0)

    # Mean reversion score: how far from equilibrium
    mr_rsi = -(rsi - 50) / 50  # RSI < 50 = positive, > 50 = negative
    mr_bb = -(bb - 0.5) * 2     # Below mid BB = positive
    mr_stoch = -(ind["stoch_k"] - 50) / 50

    raw = (0.40 * mr_rsi + 0.35 * mr_bb + 0.25 * mr_stoch) * 80

    # Trend filter: only mean-revert in established trends
    raw = raw * (0.5 + 0.5 * above_200)  # Stronger when in uptrend
    # Suppress mean-reversion buys in downtrends (catching falling knives)
    raw = np.where((raw > 0) & (above_200 < 0.5), raw * 0.3, raw)
    raw = pd.Series(raw, index=close.index)

    return raw.clip(-100, 100).ewm(span=5, adjust=False).mean()


def v15_momentum_persistence(ind):
    """v15: Momentum persistence — only signal when momentum is consistent across all TFs."""
    close = ind["close"]
    m5 = ind["mom_5"]; m10 = ind["mom_10"]; m20 = ind["mom_20"]; m40 = ind["mom_40"]

    # All 4 TFs must agree on direction
    signs = pd.concat([np.sign(m5), np.sign(m10), np.sign(m20), np.sign(m40)], axis=1)
    total_agree = signs.sum(axis=1)  # -4 to +4
    agree_pct = total_agree / 4  # -1 to +1

    # Strength = average magnitude when agreeing
    avg_mag = (m5.abs() + m10.abs() + m20.abs() + m40.abs()) / 4
    raw = agree_pct * avg_mag * 120

    # Volume confirmation
    raw = raw * (0.7 + 0.3 * (np.sign(raw) * np.sign(ind["vol_flow"])).clip(0, 1))
    raw = raw + 1  # Small drift bias
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v16_volatility_breakout(ind):
    """v16: Bollinger squeeze breakout — signal when vol compresses then expands."""
    close = ind["close"]
    squeeze = ind["bb_squeeze"]
    bb = ind["bb_pctb"]

    # Squeeze detection: low BB width = coiled spring
    is_squeezed = (squeeze < 0.2).astype(float)
    was_squeezed = is_squeezed.rolling(5).max()  # Was squeezed recently

    # Breakout direction
    breakout_up = (bb > 1.0).astype(float) * was_squeezed
    breakout_dn = (bb < 0.0).astype(float) * was_squeezed

    raw = (breakout_up - breakout_dn) * 60
    # Add trend context
    raw = raw + ind["trend_score"] * 30 + ind["mom_score"] * 20
    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v17_mfi_divergence(ind):
    """v17: Money Flow Index based — institutional flow proxy."""
    close = ind["close"]; mfi = ind["mfi"]

    # MFI signal: normalize to -1..+1
    mfi_sig = (mfi - 50) / 50

    # MFI divergence: price making new highs but MFI not
    price_high_20 = (close >= close.rolling(20).max() * 0.98).astype(float)
    mfi_not_high = (mfi < mfi.rolling(20).max() * 0.90).astype(float)
    bearish_div = price_high_20 * mfi_not_high

    price_low_20 = (close <= close.rolling(20).min() * 1.02).astype(float)
    mfi_not_low = (mfi > mfi.rolling(20).min() * 1.10).astype(float)
    bullish_div = price_low_20 * mfi_not_low

    raw = mfi_sig * 50 + bullish_div * 25 - bearish_div * 25
    raw = raw + ind["above_200"].fillna(0) * 5
    return raw.clip(-100, 100).ewm(span=4, adjust=False).mean()


def v18_cci_williams(ind):
    """v18: CCI + Williams %R combo — momentum oscillator blend."""
    close = ind["close"]
    cci = ind["cci"]; willr = ind["willr"]

    cci_n = (cci / 200).clip(-1, 1)  # Normalize CCI
    willr_n = (willr + 50) / 50  # Normalize Williams %R to -1..+1

    # Oversold bounce + overbought exhaustion
    cci_buy = pd.Series(np.where(cci < -100, (-100 - cci) / 200, 0.0), index=close.index).clip(0, 1)
    cci_sell = pd.Series(np.where(cci > 100, (cci - 100) / 200, 0.0), index=close.index).clip(0, 1)

    raw = (0.4 * cci_n + 0.3 * willr_n + 0.15 * ind["mom_score"] + 0.15 * ind["vol_flow"]) * 80
    raw = raw + cci_buy * ind["above_200"].fillna(0) * 20  # Oversold in uptrend
    raw = raw - cci_sell * (ind["ma50_slope"] < 0).astype(float) * 15
    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v19_ichimoku_trend(ind):
    """v19: Ichimoku cloud signals — above/below cloud + TK cross."""
    close = ind["close"]
    ichi_tk = ind["ichi_tk"]
    above_cloud = ind["ichi_above_cloud"]

    # TK cross signal
    tk_sig = (ichi_tk / 2).clip(-1, 1)
    cloud_sig = above_cloud * 2 - 1  # +1 above, -1 below

    raw = (0.4 * cloud_sig + 0.3 * tk_sig + 0.2 * ind["trend_score"] + 0.1 * ind["vol_flow"]) * 70
    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=4, adjust=False).mean()


def v20_hurst_adaptive(ind):
    """v20: Hurst exponent adaptive — mean-revert when H<0.5, trend-follow when H>0.5."""
    close = ind["close"]; hurst = ind["hurst"].fillna(0.5)

    # Trending regime: use momentum
    trend_sig = ind["trend_score"] * 0.5 + ind["mom_score"] * 0.5
    # Mean-revert regime: use oscillators
    mr_sig = -(ind["rsi"] - 50) / 50 * 0.5 + -(ind["bb_pctb"] - 0.5) * 0.5

    # Blend based on Hurst
    h_weight = ((hurst - 0.5) / 0.3).clip(-1, 1)  # -1 = mean-revert, +1 = trending
    trend_w = (1 + h_weight) / 2  # 0 to 1
    mr_w = 1 - trend_w

    raw = (trend_w * trend_sig + mr_w * mr_sig) * 80
    raw = raw + ind["vol_flow"] * 15 + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


def v21_efficiency_filter(ind):
    """v21: Kaufman efficiency ratio as primary filter — only trade efficient moves."""
    close = ind["close"]
    eff = ind["efficiency"].fillna(0.5)

    # Base signal: v8 core
    base = v8_baseline(ind)

    # Scale by efficiency: efficient markets get full signal, choppy markets get dampened
    eff_mult = (eff - 0.3).clip(0, 0.7) / 0.7  # 0 when eff<0.3, 1 when eff>1.0
    result = base * (0.3 + 0.7 * eff_mult)
    return result.clip(-100, 100)


def v22_structure_based(ind):
    """v22: Price structure — higher highs/higher lows pattern recognition."""
    close = ind["close"]
    hl = ind["hl_ratio"]  # Net higher highs - lower lows
    donch = ind["donch_pct"]  # Position in Donchian channel

    # Structure score: trending = HH > LL
    struct_n = (hl / 5).clip(-1, 1)

    # Donchian position as momentum proxy
    donch_sig = (donch - 0.5) * 2  # -1 to +1

    raw = (0.35 * struct_n + 0.25 * donch_sig + 0.25 * ind["trend_score"] + 0.15 * ind["vol_flow"]) * 80
    # Corrections from v8
    rsi = ind["rsi"]; above_200 = ind["above_200"].fillna(0)
    oversold = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
    raw = raw + oversold * above_200 * 30
    raw = raw + 1
    return raw.clip(-100, 100).ewm(span=3, adjust=False).mean()


# ═══════════════════════════════════════════════════════════
# MAIN: Run Batch 1
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 110)
    print("  CSI MEGA HARNESS — Batch 1: v13-v22 (10 diverse strategies)")
    print("  Testing on 101 assets (53 original + 48 additional popular stocks)")
    print("=" * 110)

    t0 = time.time()
    n_loaded = preload_all()
    print(f"\n  Loaded {n_loaded} assets in {time.time()-t0:.1f}s")
    print(f"\n  Computing indicators...")
    t1 = time.time()
    for sym in UNIVERSE:
        get_indicators(sym)
    print(f"  Indicators computed in {time.time()-t1:.1f}s")

    print(f"\n  {'Strategy':<30} {'Sharpe':>6} {'Spread':>6} {'Comb':>6} "
          f"{'BuyP':>7} {'SellC':>7} {'SH%':>5} {'GoodSep':>10} {'Mkt':>4} {'DD':>6}")
    print(f"  {'─'*30} {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*5} {'─'*10} {'─'*4} {'─'*6}")

    strategies = [
        ("v8+1 BASELINE", v8_baseline),
        ("v13 Kalman Trend", v13_kalman_trend),
        ("v14 Mean Reversion", v14_mean_reversion),
        ("v15 Mom Persistence", v15_momentum_persistence),
        ("v16 Vol Breakout", v16_volatility_breakout),
        ("v17 MFI Divergence", v17_mfi_divergence),
        ("v18 CCI+Williams", v18_cci_williams),
        ("v19 Ichimoku Trend", v19_ichimoku_trend),
        ("v20 Hurst Adaptive", v20_hurst_adaptive),
        ("v21 Efficiency Filter", v21_efficiency_filter),
        ("v22 Structure Based", v22_structure_based),
    ]

    all_results = []
    for name, fn in strategies:
        t = time.time()
        s = run_strategy(name, fn)
        s["time"] = round(time.time() - t, 1)
        all_results.append(s)

    # Leaderboard
    print(f"\n\n  {'='*80}")
    print(f"  BATCH 1 LEADERBOARD (sorted by Combined = Sharpe * (1 + Spread/100))")
    print(f"  {'='*80}")
    ranked = sorted(all_results, key=lambda x: x.get("combined", 0) if not np.isnan(x.get("combined", 0)) else -999, reverse=True)
    for i, s in enumerate(ranked):
        medal = " ** NEW CHAMPION **" if i == 0 and s["name"] != "v8+1 BASELINE" else ""
        print(f"  #{i+1:2d}: {s['name']:<28} C={s['combined']:6.3f}  Sh={s['med_sh']:6.3f}  Spr={s['spread']:+5.1f}%  "
              f"GS={s.get('good_sep','?')}{medal}")

    # Save leaderboard
    lb = []
    for s in ranked:
        entry = {k: v for k, v in s.items() if not isinstance(v, float) or not np.isnan(v)}
        lb.append(entry)
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(lb, f, indent=2, default=str)
    print(f"\n  Leaderboard saved to {LEADERBOARD_FILE}")

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.0f}s")
    print()


if __name__ == "__main__":
    main()
