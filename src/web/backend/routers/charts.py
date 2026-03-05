"""
Charts router — OHLCV data, indicators, forecasts, generated images.
"""

from typing import Optional

from fastapi import APIRouter, Query

from web.backend.services.chart_service import (
    get_available_chart_symbols,
    get_ohlcv,
    compute_indicators,
    get_generated_chart_images,
    get_forecast_data,
    get_symbols_by_sector,
)

router = APIRouter()


@router.get("/symbols")
async def chart_symbols():
    """List available symbols for charting."""
    symbols = get_available_chart_symbols()
    return {"symbols": symbols, "count": len(symbols)}


@router.get("/symbols-by-sector")
async def chart_symbols_by_sector():
    """Available symbols grouped by consolidated sector."""
    sectors = get_symbols_by_sector()
    return {"sectors": sectors, "total_sectors": len(sectors)}


@router.get("/ohlcv/{symbol}")
async def chart_ohlcv(symbol: str, tail: int = Query(default=365, le=2000)):
    """
    OHLCV candlestick data for TradingView Lightweight Charts.
    
    Returns {time, open, high, low, close, volume} series.
    """
    data = get_ohlcv(symbol, tail=tail)
    if data is None:
        return {"error": f"No price data for {symbol}"}
    return {"symbol": symbol, "data": data, "count": len(data)}


@router.get("/indicators/{symbol}")
async def chart_indicators(symbol: str, tail: int = Query(default=365, le=2000)):
    """
    Technical indicators: SMA, Bollinger, RSI, ATR.
    """
    indicators = compute_indicators(symbol, tail=tail)
    if indicators is None:
        return {"error": f"Could not compute indicators for {symbol}"}
    return {"symbol": symbol, "indicators": indicators}


@router.get("/forecast/{symbol}")
async def chart_forecast(symbol: str):
    """Multi-horizon forecast data from signal cache."""
    forecast = get_forecast_data(symbol)
    if forecast is None:
        return {"error": f"No forecast data for {symbol}"}
    return forecast


@router.get("/images")
async def chart_images():
    """List pre-generated chart PNG images."""
    images = get_generated_chart_images()
    return {"images": images, "count": len(images)}
