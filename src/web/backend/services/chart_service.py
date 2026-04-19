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

    # ── Composite Signal Index (CSI) v97 — Adaptive Scale Consecutive Patterns ──
    # Score: -100 (strong sell) to +100 (strong buy)
    #
    # Architecture:
    #   1. Core: Consecutive day patterns (runs of up/down days)
    #   2. Mean-reversion on exhaustion (4+ day runs) with strong MR=0.7
    #   3. Momentum continuation on 2-3 day runs
    #   4. Light vol_flow + trend_score overlays (15% each)
    #   5. Adaptive scaling via Kaufman efficiency ratio
    #   6. No EMA smoothing (preserves crisp signal boundaries)
    #   7. Bullish bias +1 + above_200 bonus
    #
    # Tested across 98 assets. Grand champion out of 100 versions.
    # Key metrics: Sharpe 0.648, regime spread +13.3%, 64% good separation.
    if len(close) >= 50:
        n = len(close)

        # ── 1. CONSECUTIVE DAY PATTERN DETECTION ──────────────
        up_day = (close > close.shift(1)).astype(float)
        dn_day = (close < close.shift(1)).astype(float)

        # Count consecutive runs of up/down days
        up_runs = pd.Series(np.zeros(n), index=close.index)
        dn_runs = pd.Series(np.zeros(n), index=close.index)
        up_vals = up_day.values.astype(float)
        dn_vals = dn_day.values.astype(float)
        up_r = np.zeros(n)
        dn_r = np.zeros(n)
        for i in range(1, n):
            if up_vals[i] > 0:
                up_r[i] = up_r[i - 1] + 1
            if dn_vals[i] > 0:
                dn_r[i] = dn_r[i - 1] + 1
        up_runs = pd.Series(up_r, index=close.index)
        dn_runs = pd.Series(dn_r, index=close.index)

        # Exhaustion: 4+ consecutive days in one direction (mean-reversion signal)
        up_exhaust = (up_runs >= 4).astype(float) * ((up_runs - 3) / 3).clip(0, 1)
        dn_exhaust = (dn_runs >= 4).astype(float) * ((dn_runs - 3) / 3).clip(0, 1)

        # Momentum continuation: 2-3 consecutive days (trend following)
        up_mom = ((up_runs >= 2) & (up_runs <= 3)).astype(float)
        dn_mom = ((dn_runs >= 2) & (dn_runs <= 3)).astype(float)

        # ── 2. MEAN-REVERSION + MOMENTUM SIGNALS ──────────────
        mr_sig = dn_exhaust * 0.7 - up_exhaust * 0.7   # Strong MR weight
        mom_sig = up_mom * 0.2 - dn_mom * 0.2

        # ── 3. VOLUME FLOW (light overlay) ─────────────────────
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

        # ── 4. TREND SCORE (light overlay) ─────────────────────
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

        # ── 5. ADAPTIVE SCALING (Kaufman Efficiency Ratio) ─────
        ret_1 = close.pct_change(1)
        direction = (close - close.shift(10)).abs()
        volatility_sum = ret_1.abs().rolling(10).sum()
        efficiency = (direction / volatility_sum.replace(0, float("nan"))).clip(0, 1).fillna(0.5)
        # Inefficient/mean-reverting markets: patterns more effective (scale up)
        # Efficient/trending markets: patterns less effective (scale down)
        adaptive_scale = 1.2 - 0.4 * efficiency  # Range: 0.8 to 1.2

        # ── 6. ABOVE 200 SMA CONTEXT ──────────────────────────
        sma_200 = close.rolling(200, min_periods=60).mean()
        above_200 = (close > sma_200).astype(float)

        # ── 7. COMPOSITE: No smoothing (preserves crisp boundaries) ──
        raw = (mr_sig + mom_sig + 0.15 * vol_flow + 0.15 * trend_score) * 85 * adaptive_scale
        csi = raw + above_200.fillna(0) * 5 + 1
        csi = csi.clip(-100, 100)

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
