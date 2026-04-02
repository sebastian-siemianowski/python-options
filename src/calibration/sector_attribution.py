"""
Story 6.6: Sector and Cap-Weighted Performance Attribution.

Classifies assets by sector and cap bucket, then attributes PnL
and computes per-group metrics for portfolio analysis.

Usage:
    from calibration.sector_attribution import compute_sector_attribution
    report = compute_sector_attribution(asset_pnl, sector_map, cap_map)
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Sector classifications
SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "CRM": "Technology", "ADBE": "Technology",
    "CRWD": "Technology", "NET": "Technology", "META": "Communication",
    "NFLX": "Communication", "DIS": "Communication", "SNAP": "Communication",
    "JPM": "Finance", "BAC": "Finance", "GS": "Finance", "MS": "Finance",
    "SCHW": "Finance", "AFRM": "Finance",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRNA": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "OXY": "Energy",
    "CAT": "Industrials", "DE": "Industrials", "BA": "Industrials",
    "UPS": "Industrials", "GE": "Industrials",
    "LMT": "Defence", "RTX": "Defence", "NOC": "Defence", "GD": "Defence",
    "AMZN": "Consumer", "TSLA": "Consumer", "HD": "Consumer",
    "NKE": "Consumer", "SBUX": "Consumer", "PG": "Consumer",
    "KO": "Consumer", "PEP": "Consumer", "COST": "Consumer",
    "LIN": "Materials", "FCX": "Materials", "NEM": "Materials", "NUE": "Materials",
    "SPY": "Index", "QQQ": "Index", "IWM": "Index",
}

# Cap bucket classification (simplified)
CAP_BUCKET_MAP = {
    "UPST": "Small", "AFRM": "Small", "IONQ": "Small", "SNAP": "Small",
    "CRWD": "Mid", "DKNG": "Mid", "NET": "Mid",
    "AAPL": "Large", "MSFT": "Large", "NVDA": "Large", "GOOGL": "Large",
    "AMZN": "Large", "TSLA": "Large", "META": "Large", "JPM": "Large",
    "SPY": "Index", "QQQ": "Index", "IWM": "Index",
}


@dataclass
class GroupMetrics:
    """Metrics for a sector or cap group."""
    group: str
    sharpe: float = 0.0
    hit_rate: float = 0.0
    max_drawdown: float = 0.0
    pnl_contribution: float = 0.0
    n_assets: int = 0
    avg_position_size: float = 0.0


@dataclass
class AttributionReport:
    """Full sector/cap attribution report."""
    by_sector: Dict[str, GroupMetrics] = field(default_factory=dict)
    by_cap: Dict[str, GroupMetrics] = field(default_factory=dict)
    total_pnl: float = 0.0


def compute_sector_attribution(
    asset_pnl: Dict[str, np.ndarray],
    sector_map: Optional[Dict[str, str]] = None,
    cap_map: Optional[Dict[str, str]] = None,
) -> AttributionReport:
    """
    Compute performance attribution by sector and market cap.
    
    Args:
        asset_pnl: {symbol: daily_pnl_array}.
        sector_map: {symbol: sector_name}. Defaults to built-in map.
        cap_map: {symbol: cap_bucket}. Defaults to built-in map.
    
    Returns:
        AttributionReport with per-group metrics.
    """
    if sector_map is None:
        sector_map = dict(SECTOR_MAP)
    if cap_map is None:
        cap_map = dict(CAP_BUCKET_MAP)
    
    report = AttributionReport()
    
    # Total PnL
    all_pnl = sum(float(np.sum(v)) for v in asset_pnl.values())
    report.total_pnl = all_pnl
    
    # Group by sector
    report.by_sector = _group_metrics(asset_pnl, sector_map, all_pnl)
    
    # Group by cap
    report.by_cap = _group_metrics(asset_pnl, cap_map, all_pnl)
    
    return report


def _group_metrics(
    asset_pnl: Dict[str, np.ndarray],
    grouping: Dict[str, str],
    total_pnl: float,
) -> Dict[str, GroupMetrics]:
    """Compute metrics for a grouping."""
    groups: Dict[str, List[np.ndarray]] = {}
    
    for symbol, pnl in asset_pnl.items():
        group = grouping.get(symbol, "Other")
        if group not in groups:
            groups[group] = []
        groups[group].append(np.asarray(pnl, dtype=float))
    
    result = {}
    for group, pnl_arrays in groups.items():
        # Aggregate PnL (sum across assets per day, pad to max length)
        max_len = max(len(p) for p in pnl_arrays)
        agg_pnl = np.zeros(max_len)
        for p in pnl_arrays:
            agg_pnl[:len(p)] += p
        
        gm = GroupMetrics(group=group, n_assets=len(pnl_arrays))
        gm.pnl_contribution = float(np.sum(agg_pnl))
        
        if max_len > 1:
            mean_pnl = np.mean(agg_pnl)
            std_pnl = np.std(agg_pnl, ddof=1)
            
            if std_pnl > 0:
                gm.sharpe = float((mean_pnl / std_pnl) * math.sqrt(252))
            
            gm.hit_rate = float(np.mean(agg_pnl > 0))
            
            cum = np.cumsum(agg_pnl)
            peak = np.maximum.accumulate(cum)
            dd = cum - peak
            gm.max_drawdown = float(np.min(dd))
        
        result[group] = gm
    
    return result
