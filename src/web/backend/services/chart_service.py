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
