"""
Story 5.7: Portfolio Impact Calculation (backend).

Computes portfolio-level metrics from individual asset signals:
expected return, portfolio vol, Sharpe, sector exposure, risk decomposition.

Usage:
    from decision.portfolio_impact import compute_portfolio_impact
    impact = compute_portfolio_impact(signals, allocations)
"""
import math
from typing import Dict, List, Optional


# Sector concentration warning threshold
SECTOR_CONCENTRATION_WARN = 0.40


def compute_portfolio_impact(
    signals: Dict[str, dict],
    allocations: Optional[Dict[str, float]] = None,
    horizon: int = 30,
) -> dict:
    """
    Compute portfolio-level impact from individual asset signals.
    
    Args:
        signals: {symbol: {horizon_forecasts: {H: {point_forecast_pct, ...}}, sector: str, ...}}
        allocations: {symbol: weight}. If None, equal-weight.
        horizon: Which horizon to analyze (default 30d).
    
    Returns:
        {
            expected_return_pct: float,
            portfolio_vol_pct: float,
            sharpe_estimate: float,
            sector_exposure: {sector: weight},
            risk_decomposition: [{symbol, weight, contribution_pct}],
            concentration_warnings: [str],
            asset_count: int,
        }
    """
    symbols = list(signals.keys())
    n = len(symbols)
    
    if n == 0:
        return _empty_impact()
    
    # Equal weight if not specified
    if allocations is None:
        w = 1.0 / n
        allocations = {s: w for s in symbols}
    
    # Normalize weights
    total_w = sum(allocations.values())
    if total_w > 0:
        allocations = {s: v / total_w for s, v in allocations.items()}
    
    # Extract forecasts and compute weighted return
    expected_return = 0.0
    sector_exposure: Dict[str, float] = {}
    risk_decomp: List[dict] = []
    
    for symbol in symbols:
        sig = signals[symbol]
        weight = allocations.get(symbol, 0.0)
        
        # Get forecast for the horizon
        hf = sig.get("horizon_forecasts", {})
        horizon_data = hf.get(str(horizon), hf.get(horizon, {}))
        
        if isinstance(horizon_data, (int, float)):
            fc = float(horizon_data)
        elif isinstance(horizon_data, dict):
            fc = horizon_data.get("point_forecast_pct", 0.0)
        else:
            fc = 0.0
        
        expected_return += weight * fc
        
        # Sector exposure
        sector = sig.get("sector", "Unknown")
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + weight
        
        # Risk decomposition (simplified: contribution = weight * |forecast|)
        risk_decomp.append({
            "symbol": symbol,
            "weight": weight,
            "contribution_pct": abs(weight * fc),
        })
    
    # Sort risk by contribution descending
    risk_decomp.sort(key=lambda x: x["contribution_pct"], reverse=True)
    
    # Estimate portfolio vol (simplified: sqrt of sum of squared weighted forecasts)
    vol_sum = sum((allocations.get(s, 0) * _get_fc(signals[s], horizon)) ** 2 for s in symbols)
    portfolio_vol = math.sqrt(vol_sum) if vol_sum > 0 else 0.01
    
    # Sharpe estimate
    sharpe = expected_return / portfolio_vol if portfolio_vol > 0 else 0.0
    
    # Concentration warnings
    warnings = []
    for sector, expo in sector_exposure.items():
        if expo > SECTOR_CONCENTRATION_WARN:
            warnings.append(f"{sector}: {expo:.0%} (>{SECTOR_CONCENTRATION_WARN:.0%} threshold)")
    
    return {
        "expected_return_pct": round(expected_return, 4),
        "portfolio_vol_pct": round(portfolio_vol, 4),
        "sharpe_estimate": round(sharpe, 4),
        "sector_exposure": sector_exposure,
        "risk_decomposition": risk_decomp,
        "concentration_warnings": warnings,
        "asset_count": n,
    }


def _get_fc(sig: dict, horizon: int) -> float:
    """Extract forecast from signal dict."""
    hf = sig.get("horizon_forecasts", {})
    hd = hf.get(str(horizon), hf.get(horizon, {}))
    if isinstance(hd, (int, float)):
        return float(hd)
    if isinstance(hd, dict):
        return hd.get("point_forecast_pct", 0.0)
    return 0.0


def _empty_impact() -> dict:
    return {
        "expected_return_pct": 0.0,
        "portfolio_vol_pct": 0.0,
        "sharpe_estimate": 0.0,
        "sector_exposure": {},
        "risk_decomposition": [],
        "concentration_warnings": [],
        "asset_count": 0,
    }
