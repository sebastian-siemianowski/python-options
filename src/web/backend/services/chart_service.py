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
