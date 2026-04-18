"""
Chart service — provides OHLCV data, indicators, and forecasts for interactive charts.
"""

import os
import sys
import math
from typing import Any, Dict, List, Optional, Tuple

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(SRC_DIR, "data")
PRICES_DIR = os.path.join(DATA_DIR, "prices")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")

# Import SECTOR_MAP lazily to avoid import issues at startup
_sector_map_cache: Optional[Dict[str, Any]] = None


def _get_sector_map() -> Dict[str, set]:
    """Load SECTOR_MAP from ingestion.data_utils, with caching."""
    global _sector_map_cache
    if _sector_map_cache is not None:
        return _sector_map_cache
    try:
        from ingestion.data_utils import SECTOR_MAP
        _sector_map_cache = SECTOR_MAP
        return _sector_map_cache
    except ImportError:
        return {}


# Consolidation mapping (mirrors signal_service.py)
_SECTOR_CONSOLIDATION = {
    "Indices / Broad ETFs": "Indices & ETFs",
    "Indices": "Indices & ETFs",
    "Nuclear": "Critical Materials & Nuclear",
    "Critical Materials": "Critical Materials & Nuclear",
    "AI Utility / Infrastructure": "AI & Infrastructure",
    "AI Software / Data Platforms": "AI & Infrastructure",
    "AI Hardware / Edge Compute": "AI & Infrastructure",
    "AI Power Semiconductors": "AI & Infrastructure",
    "Semiconductor Equipment": "Semiconductors",
    "FX / Commodities / Crypto": "FX, Commodities & Crypto",
}


def get_symbols_by_sector() -> List[Dict[str, Any]]:
    """
    Group available chart symbols by consolidated sector.
    
    Returns list of {name, symbols: [...]} dicts.
    """
    available = set(get_available_chart_symbols())
    sector_map = _get_sector_map()

    # Build reverse mapping: symbol -> consolidated sector
    sym_to_sector: Dict[str, str] = {}
    for sector_name, tickers in sector_map.items():
        consolidated = _SECTOR_CONSOLIDATION.get(sector_name, sector_name)
        for ticker in tickers:
            # Symbols in price files may differ from SECTOR_MAP (no =X, ^, etc.)
            # Try matching with available set
            if ticker in available:
                sym_to_sector[ticker] = consolidated

    # Group available symbols by sector
    groups: Dict[str, List[str]] = {}
    assigned = set()
    for sym, sector in sym_to_sector.items():
        groups.setdefault(sector, []).append(sym)
        assigned.add(sym)

    # Unassigned symbols go to "Other"
    unassigned = sorted(available - assigned)
    if unassigned:
        groups["Other"] = unassigned

    # Sort sectors and their symbol lists
    result = []
    for name in sorted(groups.keys()):
        result.append({
            "name": name,
            "symbols": sorted(groups[name]),
            "count": len(groups[name]),
        })
    return result


def get_available_chart_symbols() -> List[str]:
    """Return list of symbols that have price data for charting."""
    if not os.path.isdir(PRICES_DIR):
        return []
    symbols = []
    for f in sorted(os.listdir(PRICES_DIR)):
        if f.endswith(".csv"):
            sym = f.replace("_1d.csv", "").replace(".csv", "")
            symbols.append(sym)
    return symbols


def get_ohlcv(symbol: str, tail: int = 365) -> Optional[List[Dict[str, Any]]]:
    """
    Get OHLCV candlestick data for a symbol.
    
    Returns list of {time, open, high, low, close, volume} formatted
    for TradingView Lightweight Charts.
    """
    from web.backend.services.data_service import get_price_data
    data = get_price_data(symbol, tail=tail)
    if data is None:
        return None

    # Rename 'date' to 'time' for TradingView format
    return [
        {
            "time": d["date"],
            "open": d["open"],
            "high": d["high"],
            "low": d["low"],
            "close": d["close"],
            "volume": d["volume"],
        }
        for d in data
    ]


def compute_indicators(symbol: str, tail: int = 365) -> Optional[Dict[str, Any]]:
    """
    Compute technical indicators for a symbol.
    
    Returns dict with sma20, sma50, sma200, bollinger, rsi, atr series.
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        return None

    from web.backend.services.data_service import get_price_data
    raw = get_price_data(symbol, tail=tail + 200)  # Extra for warmup
    if raw is None or len(raw) < 50:
        return None

    df = pd.DataFrame(raw)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    result = {}

    # SMAs
    for period in [20, 50, 200]:
        if len(close) >= period:
            sma = close.rolling(period).mean()
            result[f"sma{period}"] = [
                {"time": df.iloc[i]["date"], "value": round(v, 4)}
                for i, v in enumerate(sma)
                if not (isinstance(v, float) and math.isnan(v))
            ]

    # Bollinger Bands (20, 2)
    if len(close) >= 20:
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        result["bollinger"] = {
            "upper": [
                {"time": df.iloc[i]["date"], "value": round(v, 4)}
                for i, v in enumerate(bb_upper)
                if not (isinstance(v, float) and math.isnan(v))
            ],
            "lower": [
                {"time": df.iloc[i]["date"], "value": round(v, 4)}
                for i, v in enumerate(bb_lower)
                if not (isinstance(v, float) and math.isnan(v))
            ],
        }

    # RSI (14)
    if len(close) >= 15:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))
        result["rsi"] = [
            {"time": df.iloc[i]["date"], "value": round(v, 2)}
            for i, v in enumerate(rsi)
            if not (isinstance(v, float) and math.isnan(v))
        ]

    # ATR (14)
    if len(close) >= 15:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        result["atr"] = [
            {"time": df.iloc[i]["date"], "value": round(v, 4)}
            for i, v in enumerate(atr)
            if not (isinstance(v, float) and math.isnan(v))
        ]

    # MACD (12, 26, 9)
    if len(close) >= 35:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        valid = ~(macd_line.isna() | signal_line.isna())
        result["macd"] = {
            "macd": [
                {"time": df.iloc[i]["date"], "value": round(v, 4)}
                for i, v in enumerate(macd_line) if valid.iloc[i]
            ],
            "signal": [
                {"time": df.iloc[i]["date"], "value": round(v, 4)}
                for i, v in enumerate(signal_line) if valid.iloc[i]
            ],
            "histogram": [
                {"time": df.iloc[i]["date"], "value": round(v, 4)}
                for i, v in enumerate(histogram) if valid.iloc[i]
            ],
        }

    # Stochastic Oscillator (14, 3, 3)
    if len(close) >= 17:
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        raw_k = 100 * (close - low14) / (high14 - low14).replace(0, float("nan"))
        k_line = raw_k.rolling(3).mean()
        d_line = k_line.rolling(3).mean()
        valid = ~(k_line.isna() | d_line.isna())
        result["stochastic"] = {
            "k": [
                {"time": df.iloc[i]["date"], "value": round(v, 2)}
                for i, v in enumerate(k_line) if valid.iloc[i]
            ],
            "d": [
                {"time": df.iloc[i]["date"], "value": round(v, 2)}
                for i, v in enumerate(d_line) if valid.iloc[i]
            ],
        }

    # ADX (14) with +DI / -DI
    if len(close) >= 28:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        # Zero out when the other is larger
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0
        atr14 = tr.ewm(span=14, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, float("nan"))
        minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, float("nan"))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
        adx = dx.ewm(span=14, adjust=False).mean()
        valid = ~(adx.isna() | plus_di.isna() | minus_di.isna())
        result["adx"] = {
            "adx": [
                {"time": df.iloc[i]["date"], "value": round(v, 2)}
                for i, v in enumerate(adx) if valid.iloc[i]
            ],
            "plus_di": [
                {"time": df.iloc[i]["date"], "value": round(v, 2)}
                for i, v in enumerate(plus_di) if valid.iloc[i]
            ],
            "minus_di": [
                {"time": df.iloc[i]["date"], "value": round(v, 2)}
                for i, v in enumerate(minus_di) if valid.iloc[i]
            ],
        }

    # OBV (On-Balance Volume)
    volume = df["volume"].astype(float)
    if len(close) >= 2:
        direction = np.sign(close.diff()).fillna(0)
        obv = (direction * volume).cumsum()
        result["obv"] = [
            {"time": df.iloc[i]["date"], "value": round(float(v), 0)}
            for i, v in enumerate(obv)
            if not (isinstance(v, float) and math.isnan(v))
        ]

    # CCI (20) - Commodity Channel Index
    if len(close) >= 20:
        tp = (high + low + close) / 3
        tp_sma = tp.rolling(20).mean()
        tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = (tp - tp_sma) / (0.015 * tp_mad.replace(0, float("nan")))
        result["cci"] = [
            {"time": df.iloc[i]["date"], "value": round(v, 2)}
            for i, v in enumerate(cci)
            if not (isinstance(v, float) and math.isnan(v))
        ]

    # MFI (14) - Money Flow Index
    if len(close) >= 15:
        tp = (high + low + close) / 3
        mf = tp * volume
        tp_diff = tp.diff()
        pos_mf = mf.where(tp_diff > 0, 0.0).rolling(14).sum()
        neg_mf = mf.where(tp_diff <= 0, 0.0).rolling(14).sum()
        mfr = pos_mf / neg_mf.replace(0, float("nan"))
        mfi = 100 - (100 / (1 + mfr))
        result["mfi"] = [
            {"time": df.iloc[i]["date"], "value": round(v, 2)}
            for i, v in enumerate(mfi)
            if not (isinstance(v, float) and math.isnan(v))
        ]

    # CMF (20) - Chaikin Money Flow
    if len(close) >= 20:
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, float("nan"))
        mfv = mfm * volume
        cmf = mfv.rolling(20).sum() / volume.rolling(20).sum().replace(0, float("nan"))
        result["cmf"] = [
            {"time": df.iloc[i]["date"], "value": round(v, 4)}
            for i, v in enumerate(cmf)
            if not (isinstance(v, float) and math.isnan(v))
        ]

    # ROC (12) - Rate of Change
    if len(close) >= 13:
        roc = 100 * (close - close.shift(12)) / close.shift(12).replace(0, float("nan"))
        result["roc"] = [
            {"time": df.iloc[i]["date"], "value": round(v, 2)}
            for i, v in enumerate(roc)
            if not (isinstance(v, float) and math.isnan(v))
        ]

    # Bollinger %B
    if len(close) >= 20:
        sma20_bb = close.rolling(20).mean()
        std20_bb = close.rolling(20).std()
        bb_upper = sma20_bb + 2 * std20_bb
        bb_lower = sma20_bb - 2 * std20_bb
        bb_width = bb_upper - bb_lower
        pct_b = (close - bb_lower) / bb_width.replace(0, float("nan"))
        result["bbpctb"] = [
            {"time": df.iloc[i]["date"], "value": round(v, 4)}
            for i, v in enumerate(pct_b)
            if not (isinstance(v, float) and math.isnan(v))
        ]

    # ── Composite Signal Index (CSI) — Multi-Factor Engine with Conditional Corrections ──
    # Score: -100 (strong sell) to +100 (strong buy)
    #
    # Architecture:
    #   1. v3 core: asymmetric buy/sell blend (trend, momentum, volume, oscillators)
    #   2. v8 corrections: conditional oversold flip, overbought rollover,
    #      volume divergence, breakdown accelerator
    #   3. Bias +1 for market drift capture
    #
    # Tested across 53 assets (50 stocks + GLD, SLV, BTC-USD).
    # Key metrics: Sharpe 0.531, regime spread +3.0%, 82% buy profitable,
    # 42.5% sell hit rate, 46% good regime separation.
    if len(close) >= 50:
        n = len(close)

        # ── 1. TREND CONTEXT: Moving Average Structure ─────────
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200, min_periods=60).mean()
        ema_10 = close.ewm(span=10, adjust=False).mean()

        # Price position relative to MAs: above = bullish context
        above_20 = (close > sma_20).astype(float)
        above_50 = (close > sma_50).astype(float)
        above_200 = (close > sma_200).astype(float)

        # MA slope (normalized): rising MA = uptrend structure
        ma20_slope = sma_20.pct_change(5)
        ma20_slope_n = (ma20_slope / ma20_slope.abs().rolling(60).max().replace(0, float("nan"))).clip(-1, 1)

        ma50_slope = sma_50.pct_change(10)
        ma50_slope_n = (ma50_slope / ma50_slope.abs().rolling(60).max().replace(0, float("nan"))).clip(-1, 1)

        # MA alignment score: +1 = price > rising 20 > rising 50, -1 = opposite
        ma_context = (0.25 * (above_20 * 2 - 1) + 0.25 * (above_50 * 2 - 1) +
                      0.25 * ma20_slope_n + 0.25 * ma50_slope_n)

        # ── 2. MULTI-TIMEFRAME MOMENTUM ────────────────────────
        ret_1 = close.pct_change(1)
        ret_5 = close.pct_change(5)
        ret_10 = close.pct_change(10)
        ret_20 = close.pct_change(20)

        # Normalize by rolling vol for comparable z-scores
        vol_20 = ret_1.rolling(20).std().replace(0, float("nan"))
        mom_fast = (ret_5 / (vol_20 * np.sqrt(5))).clip(-3, 3) / 3
        mom_med = (ret_10 / (vol_20 * np.sqrt(10))).clip(-3, 3) / 3
        mom_slow = (ret_20 / (vol_20 * np.sqrt(20))).clip(-3, 3) / 3

        # Weighted momentum with stronger recency bias
        raw_mom = 0.50 * mom_fast + 0.30 * mom_med + 0.20 * mom_slow

        # Confluence: when all TFs agree, amplify; when conflicting, dampen
        signs_mom = pd.concat([np.sign(mom_fast), np.sign(mom_med), np.sign(mom_slow)], axis=1)
        agreement = signs_mom.sum(axis=1).abs() / 3.0
        mom_score = raw_mom * (0.5 + 0.5 * agreement)

        # Momentum acceleration: is momentum increasing or decreasing?
        mom_accel = mom_fast - mom_fast.shift(5)
        mom_accel_n = (mom_accel / mom_accel.abs().rolling(30).max().replace(0, float("nan"))).clip(-1, 1)

        # ── 3. VOLUME FLOW ANALYSIS ────────────────────────────
        # Accumulation/Distribution: up-day volume vs down-day volume
        vol_sma20 = volume.rolling(20).mean().replace(0, float("nan"))
        rel_vol = (volume / vol_sma20).clip(0.1, 5.0)

        up_vol = volume.where(close > close.shift(1), 0.0).rolling(10).sum()
        dn_vol = volume.where(close <= close.shift(1), 0.0).rolling(10).sum()
        vol_ratio = (up_vol - dn_vol) / (up_vol + dn_vol).replace(0, float("nan"))
        vol_ratio = vol_ratio.clip(-1, 1)

        # On-Balance Volume trend
        obv_dir = np.sign(close.diff()).fillna(0)
        obv_raw = (obv_dir * volume).cumsum()
        obv_ema_f = obv_raw.ewm(span=10, adjust=False).mean()
        obv_ema_s = obv_raw.ewm(span=30, adjust=False).mean()
        obv_diff = obv_ema_f - obv_ema_s
        obv_range = obv_diff.abs().rolling(40).max().replace(0, float("nan"))
        obv_signal = (obv_diff / obv_range).clip(-1, 1)

        # Chaikin A/D oscillator
        clv = ((close - low) - (high - close)) / (high - low).replace(0, float("nan"))
        ad_line = (clv * volume).cumsum()
        ad_ema5 = ad_line.ewm(span=5, adjust=False).mean()
        ad_ema20 = ad_line.ewm(span=20, adjust=False).mean()
        ad_osc = ad_ema5 - ad_ema20
        ad_rng = ad_osc.abs().rolling(40).max().replace(0, float("nan"))
        ad_score = (ad_osc / ad_rng).clip(-1, 1)

        vol_flow = 0.40 * vol_ratio + 0.30 * obv_signal + 0.30 * ad_score

        # ── 4. OSCILLATOR SIGNALS (Asymmetric Buy/Sell) ────────
        # RSI
        delta_c = close.diff()
        gain_c = delta_c.clip(lower=0).rolling(14).mean()
        loss_c = (-delta_c.clip(upper=0)).rolling(14).mean()
        rs_c = gain_c / loss_c.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs_c))

        # Stochastic %K
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        stoch_k = 100 * (close - low14) / (high14 - low14).replace(0, float("nan"))
        stoch_k = stoch_k.rolling(3).mean()

        # BUY oscillator: oversold levels (RSI < 35, Stoch < 25) = buy the dip
        rsi_buy = np.where(rsi < 35, (35 - rsi) / 35, 0.0)
        stoch_buy = np.where(stoch_k < 25, (25 - stoch_k) / 25, 0.0)
        osc_buy = pd.Series(0.55 * rsi_buy + 0.45 * stoch_buy, index=close.index).clip(0, 1)

        # SELL oscillator: overbought AND declining (exhaustion pattern)
        # Not just overbought — must be overbought AND starting to roll over
        rsi_declining = (rsi < rsi.shift(3)).astype(float)
        stoch_declining = (stoch_k < stoch_k.shift(3)).astype(float)

        rsi_sell = np.where((rsi > 70) & (rsi_declining > 0), (rsi - 70) / 30, 0.0)
        stoch_sell = np.where((stoch_k > 80) & (stoch_declining > 0), (stoch_k - 80) / 20, 0.0)
        osc_sell = pd.Series(0.55 * rsi_sell + 0.45 * stoch_sell, index=close.index).clip(0, 1)

        # ── 5. MACD + ADX Trend Strength ───────────────────────
        ema12_c = close.ewm(span=12, adjust=False).mean()
        ema26_c = close.ewm(span=26, adjust=False).mean()
        macd_line_c = ema12_c - ema26_c
        signal_c = macd_line_c.ewm(span=9, adjust=False).mean()
        hist_c = macd_line_c - signal_c
        hist_range = hist_c.abs().rolling(20).max().replace(0, float("nan"))
        macd_n = (hist_c / hist_range).clip(-1, 1)

        # MACD acceleration (histogram slope)
        hist_accel = hist_c.diff(3)
        macd_accel = (hist_accel / hist_range).clip(-1, 1)

        trend_macd = 0.6 * macd_n + 0.4 * macd_accel

        # ADX + Directional
        tr_c = pd.concat([high - low, (high - close.shift()).abs(),
                          (low - close.shift()).abs()], axis=1).max(axis=1)
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

        # ── 6. VOLATILITY CONTEXT ──────────────────────────────
        vol_60 = ret_1.rolling(60).std()
        vol_pct = vol_20.rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
        # High vol → dampen confidence (noisy environment)
        vol_dampener = 1.0 - 0.3 * (vol_pct - 0.5).clip(0, 0.5)

        # ── MASTER BLEND: Asymmetric Buy/Sell Construction ─────
        # BUY SCORE: trend following + buy-the-dip + volume accumulation
        buy_raw = (
            0.30 * trend_score.clip(0, None) +       # positive trend
            0.25 * mom_score.clip(0, None) +          # positive momentum
            0.15 * osc_buy +                          # oversold bounce
            0.15 * vol_flow.clip(0, None) +           # accumulation
            0.10 * ma_context.clip(0, None) +         # above MAs
            0.05 * mom_accel_n.clip(0, None)          # accelerating up
        )

        # SELL SCORE: exhaustion + distribution + trend breakdown
        sell_raw = (
            0.25 * (-trend_score).clip(0, None) +    # negative trend
            0.25 * (-mom_score).clip(0, None) +       # negative momentum
            0.15 * osc_sell +                         # overbought exhaustion
            0.15 * (-vol_flow).clip(0, None) +        # distribution
            0.10 * (-ma_context).clip(0, None) +      # below MAs
            0.05 * (-mom_accel_n).clip(0, None) +     # accelerating down
            0.05 * (1 - above_50) * (-trend_score).clip(0, None)  # extra sell weight below SMA50
        )

        # Combine into single score
        raw_csi = (buy_raw - sell_raw) * 100

        # Apply volume confirmation: when volume agrees with signal, boost
        vol_confirm = (np.sign(raw_csi) * np.sign(vol_flow)).clip(0, 1)
        raw_csi = raw_csi * (0.80 + 0.20 * vol_confirm)

        # Apply vol dampener
        raw_csi = raw_csi * vol_dampener

        # Gate: if fewer than 2 factors agree on direction, reduce signal
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

        # Suppress weak signals: need 2+ factors agreeing
        buy_gate = (buy_factors >= 2).astype(float)
        sell_gate = (sell_factors >= 2).astype(float)
        gate = np.where(raw_csi > 0, buy_gate, np.where(raw_csi < 0, sell_gate, 0.5))
        gate = pd.Series(gate, index=close.index)
        raw_csi = raw_csi * (0.3 + 0.7 * gate)

        raw_csi = raw_csi.clip(-100, 100)

        # ── 7. ADAPTIVE SMOOTHING ──────────────────────────────
        ema_f = raw_csi.ewm(span=3, adjust=False).mean()
        ema_s = raw_csi.ewm(span=8, adjust=False).mean()
        sw = adx_regime.fillna(0.5)
        csi = sw * ema_f + (1 - sw) * ema_s

        # ── 8. CONDITIONAL CORRECTIONS (v8) ────────────────────
        # These corrections fix the #1 flaw: sell signals firing at bottoms.
        # 4 corrections target specific failure modes.

        # C1: Conditional oversold flip — THE key correction
        # In UPTREND: oversold = buying opportunity → push CSI positive
        # In DOWNTREND: oversold = genuine bear → strengthen sell signal
        oversold_c = pd.Series(np.where(rsi < 30, (30 - rsi) / 30, 0.0), index=close.index).clip(0, 1)
        stoch_os = pd.Series(np.where(stoch_k < 20, (20 - stoch_k) / 20, 0.0), index=close.index).clip(0, 1)
        os_strength = np.maximum(oversold_c, stoch_os)
        uptrend = above_200.fillna(0)
        csi = csi + os_strength * uptrend * 40
        csi = csi - os_strength * (1 - uptrend) * 10

        # C2: Overbought with SMA50 rollover — catches distribution tops
        rsi_ob = pd.Series(np.where(rsi > 68, (rsi - 68) / 32, 0.0), index=close.index).clip(0, 1)
        ma50_rolling_over = (ma50_slope < 0).astype(float)
        csi = csi - rsi_ob * ma50_rolling_over * 25
        # Overbought exhaustion: RSI was >70 recently and now declining fast
        rsi_max10 = rsi.rolling(10).max()
        rsi_was_ob = (rsi_max10 > 70).astype(float)
        rsi_declining_fast = ((rsi_max10 - rsi) / 20).clip(0, 1)
        csi = csi - rsi_was_ob * rsi_declining_fast * 12

        # C3: Volume divergence — price at highs but volume declining (smart money exiting)
        price_near_high = ((close - close.rolling(20).min()) /
                           (close.rolling(20).max() - close.rolling(20).min()).replace(0, float("nan")))
        price_near_high = price_near_high.clip(0, 1)
        vol_10 = volume.rolling(10).mean()
        vol_30 = volume.rolling(30).mean()
        vol_declining = ((vol_30 - vol_10) / vol_30.replace(0, float("nan"))).clip(0, 1)
        csi = csi - (price_near_high > 0.70).astype(float) * vol_declining * 8

        # C4: Breakdown accelerator — below SMA50 with rising volume = real selling
        below_50 = (1 - above_50).astype(float)
        vol_increasing = (vol_10 > vol_30 * 1.1).astype(float)
        neg_csi = (csi < 0).astype(float)
        csi = csi - below_50 * vol_increasing * neg_csi * 8

        # Bullish bias +1: captures market drift without destroying regime separation
        csi = csi + 1

        csi = csi.clip(-100, 100)
        csi = csi.ewm(span=2, adjust=False).mean()

        result["composite"] = [
            {"time": df.iloc[i]["date"], "value": round(float(v), 1)}
            for i, v in enumerate(csi)
            if not (math.isnan(v) if isinstance(v, float) else np.isnan(v))
        ]

    # Trim to requested tail
    for key in result:
        if isinstance(result[key], list):
            result[key] = result[key][-tail:]
        elif isinstance(result[key], dict):
            for sub in result[key]:
                result[key][sub] = result[key][sub][-tail:]

    return result


def get_generated_chart_images() -> List[Dict[str, str]]:
    """List pre-generated chart PNG images."""
    images = []
    for subdir in ["signals", "sma", "index"]:
        chart_dir = os.path.join(PLOTS_DIR, subdir)
        if not os.path.isdir(chart_dir):
            continue
        for f in sorted(os.listdir(chart_dir)):
            if f.endswith(".png"):
                images.append({
                    "filename": f,
                    "category": subdir,
                    "url": f"/static/plots/{subdir}/{f}",
                })
    return images


def get_forecast_data(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get multi-horizon forecast data from the signal cache for a symbol.
    """
    from web.backend.services.signal_service import get_asset_blocks
    blocks = get_asset_blocks()
    
    for block in blocks:
        block_symbol = block.get("symbol", "")
        if block_symbol.upper() == symbol.upper():
            signals = block.get("signals", [])
            forecasts = []
            for sig in signals:
                forecasts.append({
                    "horizon_days": sig.get("horizon_days"),
                    "expected_return_pct": sig.get("exp_ret", 0) * 100 if sig.get("exp_ret") else 0,
                    "probability_up": sig.get("p_up", 0.5),
                    "signal_label": sig.get("label", "HOLD"),
                })
            return {
                "symbol": symbol,
                "asset_label": block.get("title", symbol),
                "forecasts": forecasts,
            }
    return None
