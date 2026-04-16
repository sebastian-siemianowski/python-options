"""
Story 6.3: Per-Asset Forecast Quality Report.

Computes hit rate, MAE, IC, ECE for each asset x horizon.
Used by the profitability gate (6.4) and executive summary (6.10).

Usage:
    from calibration.forecast_quality import compute_forecast_quality
    report = compute_forecast_quality(actuals, forecasts, horizons=[1,7,30])
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


ECE_GOOD_THRESHOLD = 0.05
ECE_WARN_THRESHOLD = 0.10


@dataclass
class AssetQuality:
    """Quality metrics for one asset at one horizon."""
    asset: str
    horizon: int
    hit_rate: float = 0.0
    mae: float = 0.0
    information_coefficient: float = 0.0
    ece: float = 0.0
    n_obs: int = 0


@dataclass
class QualityReport:
    """Full forecast quality report."""
    entries: List[AssetQuality] = field(default_factory=list)
    avg_hit_rate: float = 0.0
    avg_mae: float = 0.0
    avg_ic: float = 0.0
    avg_ece: float = 0.0


def compute_forecast_quality(
    actuals: Dict[str, np.ndarray],
    forecasts: Dict[str, np.ndarray],
    horizons: Optional[List[int]] = None,
    confidence_levels: Optional[Dict[str, np.ndarray]] = None,
) -> QualityReport:
    """
    Compute per-asset, per-horizon forecast quality.
    
    Args:
        actuals: {asset: realized returns array} per horizon.
        forecasts: {asset: forecast array} per horizon.
        horizons: List of horizon labels.
        confidence_levels: {asset: predicted probabilities} for ECE.
    
    Returns:
        QualityReport with per-asset metrics.
    """
    if horizons is None:
        horizons = [1]
    
    entries: List[AssetQuality] = []
    
    for asset in sorted(actuals.keys()):
        if asset not in forecasts:
            continue
        
        a = np.asarray(actuals[asset], dtype=float)
        f = np.asarray(forecasts[asset], dtype=float)
        
        n = min(len(a), len(f))
        if n < 2:
            continue
        
        a = a[:n]
        f = f[:n]
        
        for h in horizons:
            # Hit rate: fraction where forecast sign matches actual sign
            sign_match = ((f > 0) & (a > 0)) | ((f < 0) & (a < 0))
            hr = float(np.mean(sign_match))
            
            # MAE
            mae = float(np.mean(np.abs(f - a)))
            
            # IC (rank correlation)
            ic = _rank_ic(f, a)
            
            # ECE
            conf = None
            if confidence_levels and asset in confidence_levels:
                conf = np.asarray(confidence_levels[asset], dtype=float)[:n]
            ece = _compute_ece(f, a, conf)
            
            entries.append(AssetQuality(
                asset=asset,
                horizon=h,
                hit_rate=hr,
                mae=mae,
                information_coefficient=ic,
                ece=ece,
                n_obs=n,
            ))
    
    return _aggregate_report(entries)


def _rank_ic(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Spearman rank information coefficient."""
    if len(forecasts) < 3:
        return 0.0
    
    rf = _rankdata(forecasts)
    ra = _rankdata(actuals)
    n = len(forecasts)
    d = rf - ra
    rho = 1 - (6 * np.sum(d**2)) / (n * (n**2 - 1))
    return float(rho)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Simple rank data."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    return ranks


def _compute_ece(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with optional confidence levels."""
    if confidence is None:
        # Use forecast magnitude as pseudo-confidence
        abs_fc = np.abs(forecasts)
        max_fc = np.max(abs_fc) if np.max(abs_fc) > 0 else 1.0
        confidence = abs_fc / max_fc
    
    n = len(confidence)
    if n < n_bins:
        return 0.0
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i+1])
        if i == n_bins - 1:
            mask = mask | (confidence == bin_edges[i+1])
        
        n_bin = np.sum(mask)
        if n_bin == 0:
            continue
        
        avg_conf = np.mean(confidence[mask])
        # Accuracy: fraction where sign was correct
        correct = ((forecasts[mask] > 0) & (actuals[mask] > 0)) | \
                  ((forecasts[mask] < 0) & (actuals[mask] < 0))
        avg_acc = np.mean(correct)
        
        ece += (n_bin / n) * abs(avg_acc - avg_conf)
    
    return float(ece)


def _aggregate_report(entries: List[AssetQuality]) -> QualityReport:
    """Aggregate per-entry metrics into report."""
    report = QualityReport(entries=entries)
    
    if not entries:
        return report
    
    report.avg_hit_rate = float(np.mean([e.hit_rate for e in entries]))
    report.avg_mae = float(np.mean([e.mae for e in entries]))
    report.avg_ic = float(np.mean([e.information_coefficient for e in entries]))
    report.avg_ece = float(np.mean([e.ece for e in entries]))
    
    return report
