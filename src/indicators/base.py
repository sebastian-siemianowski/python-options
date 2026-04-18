"""
Base infrastructure for the Indicators backtesting engine.
Reuses data loading and indicator computation from csi_mega_harness.py.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
PRICES_DIR = os.path.join(SRC_DIR, "data", "prices")
RESULTS_DIR = os.path.join(SRC_DIR, "data", "indicators")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
_data_cache: dict = {}


def load_ohlcv(symbol: str) -> pd.DataFrame:
    """Load OHLCV data for a symbol. Cached after first load."""
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


# ---------------------------------------------------------------------------
# Technical indicator library (computed once per symbol, cached)
# ---------------------------------------------------------------------------
_indicator_cache: dict = {}


def get_indicators(symbol: str) -> dict | None:
    """Compute 60+ technical indicators for a symbol. Returns dict or None."""
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

    ind: dict = {"close": close, "high": high, "low": low, "volume": volume, "n": n}

    # -- Moving averages --
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

    # -- MA slopes --
    ind["ma20_slope"] = ind["sma_20"].pct_change(5)
    ind["ma20_slope_n"] = (ind["ma20_slope"] / ind["ma20_slope"].abs().rolling(60).max().replace(0, np.nan)).clip(-1, 1)
    ind["ma50_slope"] = ind["sma_50"].pct_change(10)
    ind["ma50_slope_n"] = (ind["ma50_slope"] / ind["ma50_slope"].abs().rolling(60).max().replace(0, np.nan)).clip(-1, 1)

    ind["ma_context"] = (
        0.25 * (ind["above_20"] * 2 - 1) + 0.25 * (ind["above_50"] * 2 - 1) +
        0.25 * ind["ma20_slope_n"] + 0.25 * ind["ma50_slope_n"]
    )

    # -- Returns & volatility --
    ind["ret_1"] = close.pct_change(1)
    ind["vol_20"] = ind["ret_1"].rolling(20).std().replace(0, np.nan)
    ind["vol_60"] = ind["ret_1"].rolling(60).std().replace(0, np.nan)

    # -- Multi-TF momentum (vol-normalized) --
    for p in [5, 10, 20, 40]:
        ind[f"mom_{p}"] = (close.pct_change(p) / (ind["vol_20"] * np.sqrt(p))).clip(-3, 3) / 3

    raw_mom = 0.50 * ind["mom_5"] + 0.30 * ind["mom_10"] + 0.20 * ind["mom_20"]
    signs_mom = pd.concat([np.sign(ind["mom_5"]), np.sign(ind["mom_10"]), np.sign(ind["mom_20"])], axis=1)
    agreement = signs_mom.sum(axis=1).abs() / 3.0
    ind["mom_score"] = raw_mom * (0.5 + 0.5 * agreement)
    ind["mom_accel"] = ind["mom_5"] - ind["mom_5"].shift(5)
    ind["mom_accel_n"] = (ind["mom_accel"] / ind["mom_accel"].abs().rolling(30).max().replace(0, np.nan)).clip(-1, 1)

    # -- Volume flow --
    up_vol = volume.where(close > close.shift(1), 0.0).rolling(10).sum()
    dn_vol = volume.where(close <= close.shift(1), 0.0).rolling(10).sum()
    ind["vol_ratio"] = ((up_vol - dn_vol) / (up_vol + dn_vol).replace(0, np.nan)).clip(-1, 1)

    obv_dir = np.sign(close.diff()).fillna(0)
    obv_raw = (obv_dir * volume).cumsum()
    obv_ema_f = obv_raw.ewm(span=10, adjust=False).mean()
    obv_ema_s = obv_raw.ewm(span=30, adjust=False).mean()
    obv_diff = obv_ema_f - obv_ema_s
    obv_range = obv_diff.abs().rolling(40).max().replace(0, np.nan)
    ind["obv_signal"] = (obv_diff / obv_range).clip(-1, 1)

    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    ad_line = (clv * volume).cumsum()
    ad_ema5 = ad_line.ewm(span=5, adjust=False).mean()
    ad_ema20 = ad_line.ewm(span=20, adjust=False).mean()
    ad_osc = ad_ema5 - ad_ema20
    ad_rng = ad_osc.abs().rolling(40).max().replace(0, np.nan)
    ind["ad_score"] = (ad_osc / ad_rng).clip(-1, 1)
    ind["vol_flow"] = 0.40 * ind["vol_ratio"] + 0.30 * ind["obv_signal"] + 0.30 * ind["ad_score"]

    # -- RSI --
    delta_c = close.diff()
    gain_c = delta_c.clip(lower=0).rolling(14).mean()
    loss_c = (-delta_c.clip(upper=0)).rolling(14).mean()
    rs_c = gain_c / loss_c.replace(0, np.nan)
    ind["rsi"] = 100 - (100 / (1 + rs_c))

    # -- Stochastic --
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    ind["stoch_k"] = (100 * (close - low14) / (high14 - low14).replace(0, np.nan)).rolling(3).mean()

    # -- Oscillator buy/sell --
    rsi_buy = np.where(ind["rsi"] < 35, (35 - ind["rsi"]) / 35, 0.0)
    stoch_buy = np.where(ind["stoch_k"] < 25, (25 - ind["stoch_k"]) / 25, 0.0)
    ind["osc_buy"] = pd.Series(0.55 * rsi_buy + 0.45 * stoch_buy, index=close.index).clip(0, 1)

    rsi_declining = (ind["rsi"] < ind["rsi"].shift(3)).astype(float)
    stoch_declining = (ind["stoch_k"] < ind["stoch_k"].shift(3)).astype(float)
    rsi_sell = np.where((ind["rsi"] > 70) & (rsi_declining > 0), (ind["rsi"] - 70) / 30, 0.0)
    stoch_sell = np.where((ind["stoch_k"] > 80) & (stoch_declining > 0), (ind["stoch_k"] - 80) / 20, 0.0)
    ind["osc_sell"] = pd.Series(0.55 * rsi_sell + 0.45 * stoch_sell, index=close.index).clip(0, 1)

    # -- MACD --
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    hist_range = hist.abs().rolling(20).max().replace(0, np.nan)
    ind["macd_n"] = (hist / hist_range).clip(-1, 1)
    hist_accel = hist.diff(3)
    ind["macd_accel"] = (hist_accel / hist_range).clip(-1, 1)
    ind["trend_macd"] = 0.6 * ind["macd_n"] + 0.4 * ind["macd_accel"]
    ind["macd_line"] = macd_line
    ind["macd_signal"] = signal_line
    ind["macd_hist"] = hist

    # -- ADX --
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    ind["atr14"] = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / ind["atr14"].replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / ind["atr14"].replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    ind["adx"] = dx.ewm(span=14, adjust=False).mean()
    di_diff = (plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    ind["adx_regime"] = ((ind["adx"] - 15) / 35).clip(0, 1)
    ind["trend_adx"] = di_diff * ind["adx_regime"]
    ind["trend_score"] = 0.55 * ind["trend_macd"] + 0.45 * ind["trend_adx"]
    ind["plus_di"] = plus_di
    ind["minus_di"] = minus_di

    # -- Vol percentile --
    ind["vol_pct"] = ind["vol_20"].rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
    ind["vol_dampener"] = 1.0 - 0.3 * (ind["vol_pct"] - 0.5).clip(0, 0.5)

    # -- Bollinger Bands --
    bb_sma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    ind["bb_upper"] = bb_sma + 2 * bb_std
    ind["bb_lower"] = bb_sma - 2 * bb_std
    ind["bb_pctb"] = ((close - ind["bb_lower"]) / (ind["bb_upper"] - ind["bb_lower"]).replace(0, np.nan)).clip(-0.5, 1.5)
    ind["bb_width"] = (4 * bb_std / bb_sma).replace(0, np.nan)
    ind["bb_squeeze"] = ind["bb_width"].rolling(120, min_periods=20).rank(pct=True).fillna(0.5)

    # -- Keltner Channel --
    kc_mid = close.ewm(span=20, adjust=False).mean()
    kc_atr = tr.rolling(10).mean()
    ind["kc_upper"] = kc_mid + 1.5 * kc_atr
    ind["kc_lower"] = kc_mid - 1.5 * kc_atr
    ind["kc_pct"] = ((close - ind["kc_lower"]) / (ind["kc_upper"] - ind["kc_lower"]).replace(0, np.nan)).clip(-0.5, 1.5)

    # -- Williams %R --
    ind["willr"] = -100 * (high14 - close) / (high14 - low14).replace(0, np.nan)

    # -- CCI --
    tp = (high + low + close) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    ind["cci"] = ((tp - tp_sma) / (0.015 * tp_mad).replace(0, np.nan)).clip(-300, 300)

    # -- MFI --
    tp2 = (high + low + close) / 3
    mf_raw = tp2 * volume
    mf_pos = mf_raw.where(tp2 > tp2.shift(1), 0.0).rolling(14).sum()
    mf_neg = mf_raw.where(tp2 <= tp2.shift(1), 0.0).rolling(14).sum()
    mf_ratio = mf_pos / mf_neg.replace(0, np.nan)
    ind["mfi"] = 100 - (100 / (1 + mf_ratio))

    # -- Rate of Change --
    ind["roc_10"] = close.pct_change(10) * 100
    ind["roc_20"] = close.pct_change(20) * 100

    # -- ATR percent --
    ind["atr_pct"] = (ind["atr14"] / close * 100).replace(0, np.nan)

    # -- Structure --
    ind["hh"] = (high > high.rolling(20).max().shift(1)).astype(float)
    ind["ll"] = (low < low.rolling(20).min().shift(1)).astype(float)
    ind["hl_ratio"] = ind["hh"].rolling(10).sum() - ind["ll"].rolling(10).sum()

    # -- Volume relative --
    ind["vol_rel"] = (volume / volume.rolling(20).mean().replace(0, np.nan)).clip(0, 5)

    # -- Donchian Channels --
    ind["donch_high"] = high.rolling(20).max()
    ind["donch_low"] = low.rolling(20).min()
    ind["donch_pct"] = ((close - ind["donch_low"]) / (ind["donch_high"] - ind["donch_low"]).replace(0, np.nan)).clip(0, 1)

    # -- Ichimoku (simplified) --
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    ind["ichi_tk"] = ((tenkan - kijun) / close * 100).clip(-5, 5)
    ind["ichi_above_cloud"] = (close > kijun).astype(float)

    # -- Hurst exponent (simplified) --
    def _hurst_simple(s, window=100):
        vals = s.values.astype(float)
        out = np.full(len(vals), np.nan)
        log_w = np.log(window)
        for i in range(window, len(vals)):
            seg = vals[i - window:i]
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

    # -- Efficiency ratio (Kaufman) --
    direction = (close - close.shift(10)).abs()
    volatility_sum = ind["ret_1"].abs().rolling(10).sum()
    ind["efficiency"] = (direction / volatility_sum.replace(0, np.nan)).clip(0, 1)

    # -- True Range --
    ind["tr"] = tr

    # -- Open price --
    if "open" in df.columns:
        ind["open"] = df["open"].astype(float)
    else:
        ind["open"] = close.shift(1).fillna(close)

    # -- Additional convenience --
    ind["stoch_d"] = ind["stoch_k"].rolling(3).mean()

    _indicator_cache[symbol] = ind
    return ind


def clear_caches():
    """Clear all data and indicator caches."""
    _data_cache.clear()
    _indicator_cache.clear()
