"""
forecast_scorecard.py

Directional Accuracy Tracking and Scorecard (Story 1.9).

Records forecasts, matches them against realized returns, and computes
accuracy metrics per horizon per symbol.

Data flow:
  record_forecasts() -> scorecard.json (append)
  evaluate_forecasts() -> matches predictions with realized returns
  compute_scorecard_metrics() -> hit_rate, MAE, IC per horizon
  display_scorecard() -> Rich table output
"""
from __future__ import annotations

import json
import os
import sys
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

# Scorecard storage path
SCORECARD_PATH = os.path.join(REPO_SRC, "data", "calibration", "forecast_scorecard.json")
PRICES_DIR = os.path.join(REPO_SRC, "data", "prices")

# Standard horizons
HORIZONS = [1, 3, 7, 30, 90, 180, 365]


@dataclass
class ForecastRecord:
    """Single forecast point for one symbol at one horizon."""
    symbol: str
    date: str              # ISO format YYYY-MM-DD
    horizon: int           # Days ahead
    forecast_pct: float    # Predicted % change
    direction: int         # +1 or -1
    realized_pct: Optional[float] = None   # Actual % change (filled later)
    realized_direction: Optional[int] = None  # Actual direction
    evaluated: bool = False


def _load_scorecard() -> List[dict]:
    """Load existing scorecard from disk."""
    if not os.path.exists(SCORECARD_PATH):
        return []
    try:
        with open(SCORECARD_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_scorecard(records: List[dict]) -> None:
    """Save scorecard to disk."""
    os.makedirs(os.path.dirname(SCORECARD_PATH), exist_ok=True)
    with open(SCORECARD_PATH, "w") as f:
        json.dump(records, f, indent=2, default=str)


def record_forecasts(symbol: str, forecasts: List[float],
                     horizons: List[int] = None,
                     date: Optional[str] = None) -> int:
    """
    Record forecasts for a symbol into the scorecard.
    
    Args:
        symbol: Asset ticker
        forecasts: List of forecast percentages (one per horizon)
        horizons: Horizons in days (default: HORIZONS)
        date: Date string YYYY-MM-DD (default: today)
    
    Returns:
        Number of records added
    """
    if horizons is None:
        horizons = HORIZONS
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    records = _load_scorecard()
    
    # Deduplicate: skip if same (symbol, date, horizon) already exists
    existing_keys = {(r["symbol"], r["date"], r["horizon"]) for r in records}
    
    added = 0
    for h, fc in zip(horizons, forecasts):
        if (symbol, date, h) in existing_keys:
            continue
        rec = ForecastRecord(
            symbol=symbol,
            date=date,
            horizon=h,
            forecast_pct=float(fc),
            direction=1 if fc >= 0 else -1,
        )
        records.append(asdict(rec))
        added += 1
    
    if added > 0:
        _save_scorecard(records)
    
    return added


def _get_realized_return(symbol: str, start_date: str, horizon: int) -> Optional[float]:
    """
    Look up the realized return for a symbol over the given horizon.
    
    Returns percentage change or None if data not available.
    """
    try:
        price_file = os.path.join(PRICES_DIR, f"{symbol}_1d.csv")
        if not os.path.exists(price_file):
            return None
        
        df = pd.read_csv(price_file, parse_dates=["Date"], index_col="Date")
        
        start_dt = pd.Timestamp(start_date)
        end_dt = start_dt + pd.Timedelta(days=horizon)
        
        # Find nearest trading day to start and end
        if start_dt not in df.index:
            mask = df.index >= start_dt
            if not mask.any():
                return None
            start_dt = df.index[mask][0]
        
        mask_end = df.index >= end_dt
        if not mask_end.any():
            return None
        end_dt = df.index[mask_end][0]
        
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        start_price = float(df.loc[start_dt, col])
        end_price = float(df.loc[end_dt, col])
        
        if start_price <= 0:
            return None
        
        return (end_price / start_price - 1.0) * 100.0
    except Exception:
        return None


def evaluate_forecasts() -> int:
    """
    Evaluate all un-evaluated forecasts by matching with realized returns.
    
    Returns:
        Number of records evaluated.
    """
    records = _load_scorecard()
    evaluated_count = 0
    
    for rec in records:
        if rec.get("evaluated", False):
            continue
        
        realized = _get_realized_return(rec["symbol"], rec["date"], rec["horizon"])
        if realized is not None:
            rec["realized_pct"] = realized
            rec["realized_direction"] = 1 if realized >= 0 else -1
            rec["evaluated"] = True
            evaluated_count += 1
    
    if evaluated_count > 0:
        _save_scorecard(records)
    
    return evaluated_count


def compute_scorecard_metrics(symbol: Optional[str] = None) -> Dict[int, Dict[str, float]]:
    """
    Compute accuracy metrics per horizon.
    
    Args:
        symbol: If provided, filter to this symbol only
    
    Returns:
        Dict mapping horizon -> {hit_rate, mae, directional_accuracy, information_coefficient, n}
    """
    records = _load_scorecard()
    
    # Filter to evaluated records
    evaluated = [r for r in records if r.get("evaluated", False)]
    if symbol:
        evaluated = [r for r in evaluated if r["symbol"] == symbol]
    
    metrics = {}
    for h in HORIZONS:
        h_records = [r for r in evaluated if r["horizon"] == h]
        if not h_records:
            metrics[h] = {"hit_rate": 0.0, "mae": 0.0, "directional_accuracy": 0.0,
                          "information_coefficient": 0.0, "n": 0}
            continue
        
        forecasts = np.array([r["forecast_pct"] for r in h_records])
        realized = np.array([r["realized_pct"] for r in h_records])
        
        # Hit rate: direction match
        fc_dirs = np.sign(forecasts)
        re_dirs = np.sign(realized)
        hit_rate = float(np.mean(fc_dirs == re_dirs))
        
        # MAE
        mae = float(np.mean(np.abs(forecasts - realized)))
        
        # Directional accuracy (same as hit_rate but excluding zero predictions)
        nonzero = (forecasts != 0) & (realized != 0)
        if nonzero.sum() > 0:
            dir_acc = float(np.mean(np.sign(forecasts[nonzero]) == np.sign(realized[nonzero])))
        else:
            dir_acc = 0.0
        
        # Information Coefficient (rank correlation)
        if len(forecasts) >= 3:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(forecasts, realized)
            ic = float(ic) if np.isfinite(ic) else 0.0
        else:
            ic = 0.0
        
        metrics[h] = {
            "hit_rate": hit_rate,
            "mae": mae,
            "directional_accuracy": dir_acc,
            "information_coefficient": ic,
            "n": len(h_records),
        }
    
    return metrics


def display_scorecard(symbol: Optional[str] = None) -> None:
    """Display the forecast scorecard as a Rich table."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        print("Rich not available for display")
        return
    
    metrics = compute_scorecard_metrics(symbol=symbol)
    console = Console()
    
    title = f"Forecast Scorecard — {symbol}" if symbol else "Forecast Scorecard — All Assets"
    table = Table(title=title, box=box.ROUNDED, show_lines=True)
    table.add_column("Horizon", justify="right", style="bold")
    table.add_column("N", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("Dir. Acc.", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("IC", justify="right")
    
    for h in HORIZONS:
        m = metrics.get(h, {})
        n = m.get("n", 0)
        if n == 0:
            table.add_row(f"{h}D", "0", "-", "-", "-", "-")
            continue
        
        hr = m["hit_rate"]
        da = m["directional_accuracy"]
        mae = m["mae"]
        ic = m["information_coefficient"]
        
        hr_style = "bold bright_green" if hr > 0.55 else ("dim" if hr < 0.50 else "")
        ic_style = "bold bright_green" if ic > 0.1 else ("indian_red1" if ic < -0.05 else "dim")
        
        table.add_row(
            f"{h}D",
            str(n),
            f"[{hr_style}]{hr:.1%}[/{hr_style}]" if hr_style else f"{hr:.1%}",
            f"{da:.1%}",
            f"{mae:.3f}",
            f"[{ic_style}]{ic:.3f}[/{ic_style}]" if ic_style else f"{ic:.3f}",
        )
    
    console.print()
    console.print(table)
    console.print()


# =============================================================================
# Story 1.13: Per-Asset Forecast Calibration Feedback Loop
# Bayesian shrinkage amplification based on directional accuracy history.
# amp = 1.0 + (accuracy - 0.50) * shrinkage * sensitivity
# where shrinkage = n / (n + N_PRIOR)
# =============================================================================
AMP_N_PRIOR = 60         # Pseudocount for Bayesian shrinkage
AMP_SENSITIVITY = 2.0    # Maps 60% accuracy -> 1.16x at full shrinkage
AMP_MIN = 0.5            # Floor: never more than 50% dampening
AMP_MAX = 1.5            # Ceiling: never more than 50% amplification
AMP_MIN_SAMPLES = 20     # Minimum evaluated forecasts before any amplification
AMP_PRIOR_ACCURACY = 0.50  # Coin-flip prior
AMP_STALENESS_DAYS = 10  # Days before amp decays toward 1.0
AMP_STALENESS_DECAY = 0.9  # Per-day decay rate when stale


def compute_asset_amplification(accuracy: float, n: int,
                                prior_n: int = AMP_N_PRIOR) -> float:
    """
    Compute Bayesian-shrinkage amplification factor.
    
    amp = 1.0 + (accuracy - 0.50) * shrinkage_weight * sensitivity
    
    Small n -> amp ~ 1.0 (heavily shrunk toward neutral)
    Large n -> amp reflects actual accuracy
    
    Args:
        accuracy: Directional hit rate [0, 1]
        n: Number of evaluated forecast records
        prior_n: Bayesian pseudocount (default 60)
    
    Returns:
        Amplification factor in [AMP_MIN, AMP_MAX]
    """
    if n < AMP_MIN_SAMPLES:
        return 1.0
    
    shrinkage = n / (n + prior_n)
    amp = 1.0 + (accuracy - AMP_PRIOR_ACCURACY) * shrinkage * AMP_SENSITIVITY
    return float(np.clip(amp, AMP_MIN, AMP_MAX))


def get_asset_amplification(symbol: str, horizon: int = 7,
                            lookback_days: int = 60) -> float:
    """
    Look up the current amplification factor for a specific asset.
    
    Uses only past data (no leakage). Applies staleness decay
    if most recent evaluated forecast is older than AMP_STALENESS_DAYS.
    
    Args:
        symbol: Asset ticker
        horizon: Forecast horizon to use for accuracy calculation
        lookback_days: Rolling window in days for accuracy
    
    Returns:
        Amplification factor in [AMP_MIN, AMP_MAX]
    """
    records = _load_scorecard()
    
    # Filter: evaluated, matching symbol and horizon
    today = datetime.now()
    cutoff = (today - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    
    relevant = [
        r for r in records
        if r.get("evaluated", False)
        and r["symbol"] == symbol
        and r["horizon"] == horizon
        and r["date"] >= cutoff
    ]
    
    if not relevant:
        return 1.0
    
    n = len(relevant)
    
    # Directional accuracy
    hits = sum(
        1 for r in relevant
        if r.get("direction", 0) == r.get("realized_direction", 0)
        and r["direction"] != 0
    )
    accuracy = hits / max(n, 1)
    
    amp = compute_asset_amplification(accuracy, n)
    
    # Staleness decay: if most recent evaluated record is old, decay toward 1.0
    most_recent = max(r["date"] for r in relevant)
    days_stale = (today - datetime.strptime(most_recent, "%Y-%m-%d")).days
    if days_stale > AMP_STALENESS_DAYS:
        excess = days_stale - AMP_STALENESS_DAYS
        decay = AMP_STALENESS_DECAY ** excess
        amp = 1.0 + (amp - 1.0) * decay
    
    return float(np.clip(amp, AMP_MIN, AMP_MAX))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Forecast Scorecard")
    parser.add_argument("--symbol", type=str, default=None, help="Filter to specific symbol")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate pending forecasts")
    args = parser.parse_args()
    
    if args.evaluate:
        n = evaluate_forecasts()
        print(f"Evaluated {n} forecast records")
    
    display_scorecard(symbol=args.symbol)
