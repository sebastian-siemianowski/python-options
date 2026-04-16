"""
Story 2.10: Benchmark Validation Harness for Ensemble Changes.

Compares current ensemble forecasts against a stored baseline across the
12-symbol benchmark universe. Produces traffic-light summary (GREEN/AMBER/RED).

GREEN: All metrics same or better
AMBER: Degradation within tolerance
RED: Degradation beyond tolerance
"""
import os
import sys
import json
import numpy as np
from typing import Optional

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Benchmark universe (same as arena)
BENCHMARK_SYMBOLS = [
    "UPST", "AFRM", "IONQ",   # Small Cap
    "CRWD", "DKNG", "SNAP",   # Mid Cap 
    "AAPL", "NVDA", "TSLA",   # Large Cap
    "SPY", "QQQ", "IWM",      # Index
]

# Tolerance thresholds (configurable via env vars)
HIT_RATE_TOLERANCE = float(os.environ.get("VALIDATE_HIT_RATE_TOL", "0.01"))
SHARPE_TOLERANCE = float(os.environ.get("VALIDATE_SHARPE_TOL", "0.03"))
CRPS_TOLERANCE = float(os.environ.get("VALIDATE_CRPS_TOL", "0.002"))
ECE_TOLERANCE = float(os.environ.get("VALIDATE_ECE_TOL", "0.01"))

BASELINE_PATH = os.path.join(
    SCRIPT_DIR, os.pardir, "data", "calibration", "ensemble_baseline.json"
)


def compute_hit_rate(forecasts: list, realized: list) -> float:
    """Fraction of forecasts with correct sign."""
    if not forecasts or not realized:
        return 0.5
    correct = sum(
        1 for f, r in zip(forecasts, realized)
        if (f >= 0 and r >= 0) or (f < 0 and r < 0)
    )
    return correct / len(forecasts)


def compute_crps_gaussian(forecasts: list, realized: list, sigma: float = 1.0) -> float:
    """Simplified CRPS for Gaussian forecasts."""
    if not forecasts or not realized:
        return 0.0
    crps_values = []
    for f, r in zip(forecasts, realized):
        # CRPS for point forecast with assumed Gaussian spread
        z = (r - f) / max(sigma, 1e-10)
        from scipy.stats import norm
        crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        crps_values.append(abs(crps))
    return float(np.mean(crps_values))


def compute_ece(forecasts: list, realized: list, n_bins: int = 10) -> float:
    """Expected Calibration Error (simplified)."""
    if not forecasts or not realized:
        return 0.0
    # Convert to probabilities via sigmoid of forecast magnitude
    probs = [1 / (1 + np.exp(-abs(f) * 0.5)) for f in forecasts]
    correct = [1.0 if (f >= 0 and r >= 0) or (f < 0 and r < 0) else 0.0
               for f, r in zip(forecasts, realized)]
    
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    total = len(probs)
    for b in range(n_bins):
        mask = [(bin_edges[b] <= p < bin_edges[b + 1]) for p in probs]
        n_in_bin = sum(mask)
        if n_in_bin == 0:
            continue
        avg_conf = np.mean([p for p, m in zip(probs, mask) if m])
        avg_acc = np.mean([c for c, m in zip(correct, mask) if m])
        ece += abs(avg_conf - avg_acc) * (n_in_bin / total)
    return float(ece)


def compute_sharpe(forecasts: list, realized: list) -> float:
    """Sharpe-like metric: mean(sign_forecast * realized) / std(realized)."""
    if not forecasts or not realized or len(realized) < 5:
        return 0.0
    signed_returns = [
        np.sign(f) * r for f, r in zip(forecasts, realized)
    ]
    mean_ret = np.mean(signed_returns)
    std_ret = np.std(signed_returns)
    if std_ret < 1e-10:
        return 0.0
    return float(mean_ret / std_ret)


def evaluate_metrics(forecasts: list, realized: list, sigma: float = 1.0) -> dict:
    """Compute full metric suite."""
    return {
        "hit_rate": compute_hit_rate(forecasts, realized),
        "sharpe": compute_sharpe(forecasts, realized),
        "crps": compute_crps_gaussian(forecasts, realized, sigma),
        "ece": compute_ece(forecasts, realized),
    }


def load_baseline() -> Optional[dict]:
    """Load stored baseline metrics."""
    if not os.path.exists(BASELINE_PATH):
        return None
    with open(BASELINE_PATH, "r") as f:
        return json.load(f)


def save_baseline(metrics: dict):
    """Save current metrics as the new baseline."""
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def compare_metrics(current: dict, baseline: dict) -> dict:
    """
    Compare current vs baseline metrics.
    
    Returns dict with:
      - deltas: per-metric delta values
      - verdict: "GREEN", "AMBER", or "RED"
      - details: per-metric pass/fail
    """
    deltas = {
        "hit_rate": current["hit_rate"] - baseline["hit_rate"],
        "sharpe": current["sharpe"] - baseline["sharpe"],
        "crps": current["crps"] - baseline["crps"],  # lower is better
        "ece": current["ece"] - baseline["ece"],      # lower is better
    }
    
    details = {}
    any_red = False
    any_amber = False
    
    # Hit rate: higher is better
    if deltas["hit_rate"] >= 0:
        details["hit_rate"] = "GREEN"
    elif deltas["hit_rate"] >= -HIT_RATE_TOLERANCE:
        details["hit_rate"] = "AMBER"
        any_amber = True
    else:
        details["hit_rate"] = "RED"
        any_red = True
    
    # Sharpe: higher is better
    if deltas["sharpe"] >= 0:
        details["sharpe"] = "GREEN"
    elif deltas["sharpe"] >= -SHARPE_TOLERANCE:
        details["sharpe"] = "AMBER"
        any_amber = True
    else:
        details["sharpe"] = "RED"
        any_red = True
    
    # CRPS: lower is better
    if deltas["crps"] <= 0:
        details["crps"] = "GREEN"
    elif deltas["crps"] <= CRPS_TOLERANCE:
        details["crps"] = "AMBER"
        any_amber = True
    else:
        details["crps"] = "RED"
        any_red = True
    
    # ECE: lower is better
    if deltas["ece"] <= 0:
        details["ece"] = "GREEN"
    elif deltas["ece"] <= ECE_TOLERANCE:
        details["ece"] = "AMBER"
        any_amber = True
    else:
        details["ece"] = "RED"
        any_red = True
    
    if any_red:
        verdict = "RED"
    elif any_amber:
        verdict = "AMBER"
    else:
        verdict = "GREEN"
    
    return {
        "deltas": deltas,
        "verdict": verdict,
        "details": details,
    }
