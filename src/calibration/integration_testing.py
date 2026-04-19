"""
Epic 30: End-to-End Integration Testing

Three components:
1. Full pipeline smoke test (runs tune + signal validation)
2. Golden scores regression registry
3. Temporal cross-validation consistency

These functions validate the full pipeline chain:
  Prices -> Returns -> Vol -> Regimes -> Model Fits -> BMA -> Signals -> Confidence
"""

import os
import sys
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Story 30.2: Regression thresholds
BIC_REGRESSION_THRESHOLD = 20.0       # BIC worsens by > 20 nats
CRPS_REGRESSION_THRESHOLD = 0.003     # CRPS worsens by > 0.003
HIT_RATE_REGRESSION_THRESHOLD = 0.02  # Hit rate drops > 2%

# Story 30.3: Cross-validation consistency
CV_MAX_COV = 0.15   # Coefficient of variation across folds
CV_OUTLIER_SD = 2.0  # Outlier threshold in standard deviations
CV_MIN_HIT_RATE = 0.48  # Worst fold must beat random

# Default golden scores path
GOLDEN_SCORES_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR), "data", "golden_scores.json"
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PipelineSmokeResult:
    """Result of full pipeline smoke test for a single asset."""
    asset: str
    success: bool
    bic: Optional[float] = None
    signal_direction: Optional[int] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    n_models: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "success": self.success,
            "bic": self.bic,
            "signal_direction": self.signal_direction,
            "confidence": self.confidence,
            "error": self.error,
            "n_models": self.n_models,
        }


@dataclass
class RegressionTestResult:
    """Result of golden score regression test."""
    asset: str
    passed: bool
    regressions: List[str] = field(default_factory=list)
    current_bic: Optional[float] = None
    current_crps: Optional[float] = None
    current_hit_rate: Optional[float] = None
    baseline_bic: Optional[float] = None
    baseline_crps: Optional[float] = None
    baseline_hit_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "passed": self.passed,
            "regressions": self.regressions,
            "current_bic": self.current_bic,
            "current_crps": self.current_crps,
            "current_hit_rate": self.current_hit_rate,
            "baseline_bic": self.baseline_bic,
            "baseline_crps": self.baseline_crps,
            "baseline_hit_rate": self.baseline_hit_rate,
        }


@dataclass
class TemporalCVResult:
    """Result of temporal cross-validation consistency test."""
    n_folds: int
    fold_metrics: List[Dict[str, float]]  # Per-fold: bic, crps, hit_rate
    cv_bic: float = 0.0      # Coefficient of variation for BIC
    cv_crps: float = 0.0     # CoV for CRPS
    cv_hit_rate: float = 0.0 # CoV for hit rate
    all_folds_consistent: bool = False
    worst_hit_rate: float = 0.0
    outlier_folds: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "fold_metrics": self.fold_metrics,
            "cv_bic": float(self.cv_bic),
            "cv_crps": float(self.cv_crps),
            "cv_hit_rate": float(self.cv_hit_rate),
            "all_folds_consistent": self.all_folds_consistent,
            "worst_hit_rate": float(self.worst_hit_rate),
            "outlier_folds": self.outlier_folds,
        }


# ---------------------------------------------------------------------------
# Story 30.1: Full Pipeline Smoke Test
# ---------------------------------------------------------------------------

def validate_pipeline_output(
    asset: str,
    tune_result: Optional[Dict],
    signal_direction: Optional[int] = None,
    confidence: Optional[float] = None,
) -> PipelineSmokeResult:
    """
    Validate tune + signal output for a single asset.

    Parameters
    ----------
    asset : str
        Asset symbol.
    tune_result : dict or None
        Output from tune_asset_with_bma().
    signal_direction : int or None
        Signal direction from signal generation.
    confidence : float or None
        Signal confidence from signal generation.

    Returns
    -------
    PipelineSmokeResult
    """
    if tune_result is None:
        return PipelineSmokeResult(
            asset=asset, success=False, error="tune_result is None"
        )

    # Validate tune output structure
    errors = []

    # Check required keys
    for key in ("asset", "global", "meta"):
        if key not in tune_result:
            errors.append(f"Missing key: {key}")

    if errors:
        return PipelineSmokeResult(
            asset=asset, success=False, error="; ".join(errors)
        )

    # Extract BIC from global models
    bic = None
    n_models = 0
    global_data = tune_result.get("global", {})
    models = global_data.get("models", {})
    n_models = len(models)

    # Find best BIC across models
    bic_values = []
    for model_name, model_data in models.items():
        if isinstance(model_data, dict) and "bic" in model_data:
            b = model_data["bic"]
            if b is not None and np.isfinite(b):
                bic_values.append(b)

    if bic_values:
        bic = min(bic_values)  # Best (lowest) BIC

    # Validate BIC
    if bic is not None and not np.isfinite(bic):
        errors.append(f"BIC is not finite: {bic}")
        bic = None

    # Validate signal direction
    if signal_direction is not None:
        if signal_direction not in (-1, 0, 1):
            errors.append(f"Invalid signal direction: {signal_direction}")

    # Validate confidence
    if confidence is not None:
        if not (0.0 <= confidence <= 1.0):
            errors.append(f"Confidence out of range: {confidence}")
        if not np.isfinite(confidence):
            errors.append(f"Confidence not finite: {confidence}")

    success = len(errors) == 0

    return PipelineSmokeResult(
        asset=asset,
        success=success,
        bic=bic,
        signal_direction=signal_direction,
        confidence=confidence,
        error="; ".join(errors) if errors else None,
        n_models=n_models,
    )


def aggregate_smoke_results(results: List[PipelineSmokeResult]) -> Dict[str, Any]:
    """
    Aggregate smoke test results across all assets.

    Parameters
    ----------
    results : list of PipelineSmokeResult

    Returns
    -------
    dict with summary: total, passed, failed, failure_details
    """
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = [r for r in results if not r.success]

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total > 0 else 0.0,
        "all_passed": len(failed) == 0,
        "failure_details": [r.to_dict() for r in failed],
    }


# ---------------------------------------------------------------------------
# Story 30.2: Golden Score Regression Registry
# ---------------------------------------------------------------------------

def load_golden_scores(path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Load baseline golden scores from JSON file.

    Returns dict of {asset: {bic, crps, hit_rate}}.
    """
    path = path or GOLDEN_SCORES_PATH
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_golden_scores(
    scores: Dict[str, Dict[str, float]],
    path: Optional[str] = None,
) -> None:
    """Save golden scores to JSON (explicit approval only)."""
    path = path or GOLDEN_SCORES_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(scores, f, indent=2, sort_keys=True)


def check_regression(
    asset: str,
    current_bic: Optional[float],
    current_crps: Optional[float],
    current_hit_rate: Optional[float],
    golden_scores: Dict[str, Dict[str, float]],
) -> RegressionTestResult:
    """
    Check if current metrics regressed relative to golden baseline.

    Parameters
    ----------
    asset : str
    current_bic : float or None
    current_crps : float or None
    current_hit_rate : float or None
    golden_scores : dict
        {asset: {bic, crps, hit_rate}}

    Returns
    -------
    RegressionTestResult
    """
    baseline = golden_scores.get(asset)
    if baseline is None:
        # No baseline: pass (new asset)
        return RegressionTestResult(
            asset=asset,
            passed=True,
            current_bic=current_bic,
            current_crps=current_crps,
            current_hit_rate=current_hit_rate,
        )

    regressions = []
    baseline_bic = baseline.get("bic")
    baseline_crps = baseline.get("crps")
    baseline_hit_rate = baseline.get("hit_rate")

    # BIC regression: current - baseline > threshold (higher BIC = worse)
    if current_bic is not None and baseline_bic is not None:
        if current_bic - baseline_bic > BIC_REGRESSION_THRESHOLD:
            regressions.append(
                f"BIC regressed: {current_bic:.1f} vs baseline {baseline_bic:.1f} "
                f"(delta={current_bic - baseline_bic:.1f}, threshold={BIC_REGRESSION_THRESHOLD})"
            )

    # CRPS regression: current - baseline > threshold (higher CRPS = worse)
    if current_crps is not None and baseline_crps is not None:
        if current_crps - baseline_crps > CRPS_REGRESSION_THRESHOLD:
            regressions.append(
                f"CRPS regressed: {current_crps:.4f} vs baseline {baseline_crps:.4f} "
                f"(delta={current_crps - baseline_crps:.4f}, threshold={CRPS_REGRESSION_THRESHOLD})"
            )

    # Hit rate regression: baseline - current > threshold (lower hit_rate = worse)
    if current_hit_rate is not None and baseline_hit_rate is not None:
        if baseline_hit_rate - current_hit_rate > HIT_RATE_REGRESSION_THRESHOLD:
            regressions.append(
                f"Hit rate regressed: {current_hit_rate:.3f} vs baseline {baseline_hit_rate:.3f} "
                f"(delta={baseline_hit_rate - current_hit_rate:.3f}, threshold={HIT_RATE_REGRESSION_THRESHOLD})"
            )

    return RegressionTestResult(
        asset=asset,
        passed=len(regressions) == 0,
        regressions=regressions,
        current_bic=current_bic,
        current_crps=current_crps,
        current_hit_rate=current_hit_rate,
        baseline_bic=baseline_bic,
        baseline_crps=baseline_crps,
        baseline_hit_rate=baseline_hit_rate,
    )


# ---------------------------------------------------------------------------
# Story 30.3: Temporal Cross-Validation Consistency
# ---------------------------------------------------------------------------

def temporal_cv_split(
    returns: np.ndarray,
    n_folds: int = 5,
    min_train_frac: float = 0.3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window temporal cross-validation splits.

    Temporal ordering is preserved: fold k always uses earlier data than fold k+1.
    Training windows expand; test windows are non-overlapping.

    Parameters
    ----------
    returns : array-like
        Full return series.
    n_folds : int
        Number of folds.
    min_train_frac : float
        Minimum fraction of data used for training in the first fold.

    Returns
    -------
    List of (train_indices, test_indices) tuples.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    T = len(r)

    if T < 2 * n_folds:
        raise ValueError(f"Not enough data ({T}) for {n_folds} folds")

    min_train = max(int(T * min_train_frac), n_folds)
    remaining = T - min_train
    fold_size = remaining // n_folds

    if fold_size < 1:
        raise ValueError(f"Fold size too small: {fold_size}")

    splits = []
    for k in range(n_folds):
        test_start = min_train + k * fold_size
        test_end = min_train + (k + 1) * fold_size if k < n_folds - 1 else T

        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits


def temporal_cv_consistency(
    fold_metrics: List[Dict[str, float]],
    n_folds: int = 5,
) -> TemporalCVResult:
    """
    Assess consistency of metrics across temporal CV folds.

    Parameters
    ----------
    fold_metrics : list of dict
        Each dict has: bic, crps, hit_rate (per fold).
    n_folds : int
        Number of folds.

    Returns
    -------
    TemporalCVResult
    """
    if len(fold_metrics) < 2:
        return TemporalCVResult(
            n_folds=len(fold_metrics),
            fold_metrics=fold_metrics,
            all_folds_consistent=len(fold_metrics) == 1,
            worst_hit_rate=fold_metrics[0].get("hit_rate", 0) if fold_metrics else 0,
        )

    # Extract per-metric arrays
    bics = np.array([m.get("bic", np.nan) for m in fold_metrics])
    crps_vals = np.array([m.get("crps", np.nan) for m in fold_metrics])
    hit_rates = np.array([m.get("hit_rate", np.nan) for m in fold_metrics])

    def _safe_cov(arr: np.ndarray) -> float:
        """Coefficient of variation, safe for zero/nan."""
        valid = arr[np.isfinite(arr)]
        if len(valid) < 2:
            return 0.0
        mean = np.mean(valid)
        if abs(mean) < 1e-15:
            return 0.0
        return float(np.std(valid) / abs(mean))

    cv_bic = _safe_cov(bics)
    cv_crps = _safe_cov(crps_vals)
    cv_hit_rate = _safe_cov(hit_rates)

    # Detect outlier folds
    outlier_folds = []
    for arr_name, arr in [("bic", bics), ("crps", crps_vals), ("hit_rate", hit_rates)]:
        valid = arr[np.isfinite(arr)]
        if len(valid) < 2:
            continue
        mean = np.mean(valid)
        std = np.std(valid)
        if std < 1e-15:
            continue
        for i, v in enumerate(arr):
            if np.isfinite(v) and abs(v - mean) > CV_OUTLIER_SD * std:
                if i not in outlier_folds:
                    outlier_folds.append(i)

    # Worst fold hit rate
    valid_hr = hit_rates[np.isfinite(hit_rates)]
    worst_hit_rate = float(np.min(valid_hr)) if len(valid_hr) > 0 else 0.0

    # Consistency check
    all_consistent = (
        cv_bic < CV_MAX_COV
        and cv_crps < CV_MAX_COV
        and cv_hit_rate < CV_MAX_COV
        and len(outlier_folds) == 0
        and worst_hit_rate >= CV_MIN_HIT_RATE
    )

    return TemporalCVResult(
        n_folds=n_folds,
        fold_metrics=fold_metrics,
        cv_bic=cv_bic,
        cv_crps=cv_crps,
        cv_hit_rate=cv_hit_rate,
        all_folds_consistent=all_consistent,
        worst_hit_rate=worst_hit_rate,
        outlier_folds=sorted(outlier_folds),
    )
