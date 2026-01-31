#!/usr/bin/env python3
"""
pit_calibration.py

Probability Integral Transform (PIT) calibration verification for Level-7 quantitative systems.

PIT calibration is the single most important test for probabilistic forecasts because:
- Kelly sizing assumes probabilities are calibrated
- Overconfident probabilities => over-bet => ruin risk
- Underconfident probabilities => under-bet => missed opportunities

Tests:
1. Reliability diagram: predicted vs actual frequencies (should be diagonal)
2. Uniform histogram: distribution of predicted probabilities (should be uniform under H0)
3. Calibration error: Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
4. Brier score: proper scoring rule for probabilistic accuracy

Result:
- Well-calibrated: P↑=0.62 means 62% of outcomes are positive
- Overconfident: P↑=0.70 but only 55% outcomes positive => scale down confidence
- Underconfident: P↑=0.62 but 70% outcomes positive => scale up confidence
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class CalibrationMetrics:
    """Container for calibration test results."""
    n_predictions: int
    n_bins: int
    bin_edges: np.ndarray
    bin_counts: np.ndarray
    predicted_probs: np.ndarray  # mean predicted probability per bin
    actual_frequencies: np.ndarray  # actual frequency of positive outcomes per bin
    expected_calibration_error: float  # ECE: weighted mean absolute deviation
    maximum_calibration_error: float  # MCE: max absolute deviation
    brier_score: float  # mean squared error of probabilities
    uniformity_test_statistic: float  # chi-squared statistic for uniform test
    uniformity_pvalue: float  # p-value for uniformity (>0.05 => pass)
    calibrated: bool  # True if well-calibrated (ECE < threshold)
    calibration_diagnosis: str  # "well_calibrated", "overconfident", or "underconfident"


def compute_pit_calibration(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10,
    ece_threshold: float = 0.05
) -> CalibrationMetrics:
    """
    Compute PIT calibration metrics for predicted probabilities vs actual outcomes.
    
    Probability Integral Transform (PIT):
    If probabilities are well-calibrated, the distribution of predicted probabilities
    should match the empirical frequency of positive outcomes.
    
    Tests:
    1. Reliability diagram: For each bin of predicted probabilities,
       compute actual frequency of positive outcomes. Should lie on y=x diagonal.
    2. Expected Calibration Error (ECE): weighted mean absolute deviation from diagonal
    3. Maximum Calibration Error (MCE): worst-case deviation
    4. Brier score: mean squared error (proper scoring rule)
    5. Uniformity test: chi-squared test for uniform distribution of predictions
    
    Args:
        predicted_probs: Array of predicted probabilities P(outcome > 0), shape (n,)
        actual_outcomes: Array of actual outcomes (continuous), shape (n,)
                        Converted to binary: 1 if outcome > 0, else 0
        n_bins: Number of bins for reliability diagram (default 10)
        ece_threshold: ECE threshold for "well-calibrated" classification (default 0.05)
        
    Returns:
        CalibrationMetrics dataclass with all calibration diagnostics
    """
    # Input validation
    predicted_probs = np.asarray(predicted_probs, dtype=float).ravel()
    actual_outcomes = np.asarray(actual_outcomes, dtype=float).ravel()
    
    if predicted_probs.shape != actual_outcomes.shape:
        raise ValueError(f"Shape mismatch: predicted_probs {predicted_probs.shape} != actual_outcomes {actual_outcomes.shape}")
    
    # Remove NaNs/Infs
    valid_mask = np.isfinite(predicted_probs) & np.isfinite(actual_outcomes)
    predicted_probs = predicted_probs[valid_mask]
    actual_outcomes = actual_outcomes[valid_mask]
    
    n = len(predicted_probs)
    if n < 10:
        raise ValueError(f"Insufficient data: {n} predictions (need >= 10)")
    
    # Clip probabilities to valid range
    predicted_probs = np.clip(predicted_probs, 0.0, 1.0)
    
    # Convert actual outcomes to binary (1 if positive, 0 otherwise)
    actual_binary = (actual_outcomes > 0).astype(float)
    
    # Create bins for reliability diagram
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(predicted_probs, bin_edges[1:-1])  # 0 to n_bins-1
    
    # Compute statistics per bin
    bin_counts = np.zeros(n_bins, dtype=int)
    bin_predicted_mean = np.zeros(n_bins, dtype=float)
    bin_actual_freq = np.zeros(n_bins, dtype=float)
    
    for i in range(n_bins):
        mask = (bin_indices == i)
        count = np.sum(mask)
        bin_counts[i] = count
        
        if count > 0:
            bin_predicted_mean[i] = float(np.mean(predicted_probs[mask]))
            bin_actual_freq[i] = float(np.mean(actual_binary[mask]))
        else:
            # Empty bin: use bin center as predicted, NaN as actual
            bin_predicted_mean[i] = float((bin_edges[i] + bin_edges[i+1]) / 2.0)
            bin_actual_freq[i] = float('nan')
    
    # Expected Calibration Error (ECE)
    # ECE = Σ (n_i / n) * |predicted_i - actual_i|
    # Weighted by bin size to avoid empty bins dominating
    valid_bins = bin_counts > 0
    if np.sum(valid_bins) > 0:
        weights = bin_counts[valid_bins] / n
        deviations = np.abs(bin_predicted_mean[valid_bins] - bin_actual_freq[valid_bins])
        ece = float(np.sum(weights * deviations))
    else:
        ece = float('nan')
    
    # Maximum Calibration Error (MCE)
    # MCE = max |predicted_i - actual_i| over non-empty bins
    if np.sum(valid_bins) > 0:
        deviations_all = np.abs(bin_predicted_mean[valid_bins] - bin_actual_freq[valid_bins])
        mce = float(np.max(deviations_all))
    else:
        mce = float('nan')
    
    # Brier score: mean squared error of probabilities
    # BS = (1/n) * Σ (p_i - y_i)²
    # Lower is better (0 = perfect)
    brier_score = float(np.mean((predicted_probs - actual_binary) ** 2))
    
    # Uniformity test: chi-squared goodness-of-fit
    # Under H0 (well-calibrated), predicted probabilities should be uniformly distributed
    # Expected count per bin = n / n_bins
    expected_count = n / n_bins
    # Chi-squared statistic: Σ (observed - expected)² / expected
    chi2_stat = float(np.sum((bin_counts - expected_count) ** 2 / expected_count))
    
    # Degrees of freedom = n_bins - 1
    from scipy.stats import chi2
    df = n_bins - 1
    uniformity_pvalue = float(1.0 - chi2.cdf(chi2_stat, df))
    
    # Classification: well-calibrated, overconfident, or underconfident
    calibrated = (ece < ece_threshold) if np.isfinite(ece) else False
    
    # Diagnose bias direction
    if np.sum(valid_bins) > 0:
        # Average signed deviation: positive => overconfident, negative => underconfident
        signed_deviations = bin_predicted_mean[valid_bins] - bin_actual_freq[valid_bins]
        avg_bias = float(np.mean(signed_deviations))
        
        if calibrated:
            diagnosis = "well_calibrated"
        elif avg_bias > 0.02:
            diagnosis = "overconfident"
        elif avg_bias < -0.02:
            diagnosis = "underconfident"
        else:
            diagnosis = "slightly_miscalibrated"
    else:
        diagnosis = "insufficient_data"
    
    return CalibrationMetrics(
        n_predictions=int(n),
        n_bins=int(n_bins),
        bin_edges=bin_edges,
        bin_counts=bin_counts,
        predicted_probs=bin_predicted_mean,
        actual_frequencies=bin_actual_freq,
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
        brier_score=brier_score,
        uniformity_test_statistic=chi2_stat,
        uniformity_pvalue=uniformity_pvalue,
        calibrated=calibrated,
        calibration_diagnosis=diagnosis,
    )


def collect_predictions_walk_forward(
    px: pd.Series,
    horizons: List[int],
    train_days: int = 504,
    test_days: int = 21,
    max_predictions: int = 500
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Collect predicted probabilities and actual outcomes via walk-forward out-of-sample testing.
    
    This is the key function for PIT calibration: we need true out-of-sample predictions
    to avoid look-ahead bias. Walk-forward ensures each prediction uses only past data.
    
    Process:
    1. Split data into non-overlapping train/test windows
    2. Fit model on train window
    3. Predict probabilities for test window
    4. Record actual outcomes H days forward
    5. Repeat for all windows
    
    Args:
        px: Price series
        horizons: List of forecast horizons to test (days)
        train_days: Training window size (default 504 = ~2 years)
        test_days: Test window size (default 21 = ~1 month)
        max_predictions: Maximum number of predictions to collect (stops early if reached)
        
    Returns:
        Dictionary mapping horizon to {
            'predicted_probs': array of predicted P(return > 0),
            'actual_returns': array of actual returns over horizon,
            'forecast_dates': array of forecast dates
        }
    """
    from decision.signals import compute_features, latest_signals
    
    px_clean = px.dropna()
    if len(px_clean) < train_days + test_days + max(horizons):
        raise ValueError(f"Insufficient data: {len(px_clean)} observations")
    
    log_px = np.log(px_clean)
    dates = px_clean.index
    
    # Define walk-forward windows
    windows = []
    start_idx = 0
    while start_idx + train_days + test_days <= len(dates):
        train_end_idx = start_idx + train_days
        test_end_idx = min(train_end_idx + test_days, len(dates))
        
        train_dates = dates[start_idx:train_end_idx]
        test_dates = dates[train_end_idx:test_end_idx]
        
        if len(test_dates) > 0:
            windows.append({
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
            })
        
        # Move forward by test_days (non-overlapping)
        start_idx = test_end_idx
        
        # Early stop if we have enough predictions
        if len(windows) * test_days >= max_predictions:
            break
    
    if not windows:
        raise ValueError("No valid walk-forward windows")
    
    # Storage for predictions per horizon
    results = {h: {'predicted_probs': [], 'actual_returns': [], 'forecast_dates': []} for h in horizons}
    
    for window in windows:
        # Fit features on training data
        train_px = px_clean.loc[window["train_start"]:window["train_end"]]
        
        try:
            train_feats = compute_features(train_px)
            
            # Get predictions at end of training window
            last_close = float(train_px.iloc[-1])
            signals, _ = latest_signals(train_feats, horizons, last_close=last_close, t_map=True, ci=0.68)
            
            # Extract predicted probabilities per horizon
            pred_dict = {s.horizon_days: s.p_up for s in signals}
            
            # For each horizon, compute actual outcome H days forward from train_end
            train_end_log_px = float(log_px.loc[window["train_end"]])
            train_end_idx = dates.get_loc(window["train_end"])
            
            for H in horizons:
                pred_prob = pred_dict.get(H, np.nan)
                
                if not np.isfinite(pred_prob):
                    continue
                
                # Actual return H days forward
                forward_idx = train_end_idx + H
                if forward_idx < len(dates):
                    forward_date = dates[forward_idx]
                    actual_log_px = float(log_px.loc[forward_date])
                    actual_ret_H = actual_log_px - train_end_log_px
                    
                    # Store prediction and outcome
                    results[H]['predicted_probs'].append(pred_prob)
                    results[H]['actual_returns'].append(actual_ret_H)
                    results[H]['forecast_dates'].append(window["train_end"])
                    
        except Exception:
            # Skip windows where feature computation fails
            continue
    
    # Convert lists to arrays
    for H in horizons:
        results[H]['predicted_probs'] = np.array(results[H]['predicted_probs'], dtype=float)
        results[H]['actual_returns'] = np.array(results[H]['actual_returns'], dtype=float)
        results[H]['forecast_dates'] = pd.DatetimeIndex(results[H]['forecast_dates'])
    
    return results


def run_pit_calibration_test(
    px: pd.Series,
    horizons: List[int] = [1, 21, 63],
    n_bins: int = 10,
    train_days: int = 504,
    test_days: int = 21,
    max_predictions: int = 500
) -> Dict[int, CalibrationMetrics]:
    """
    Run full PIT calibration test suite for multiple horizons.
    
    This is the main entry point for calibration verification. It:
    1. Collects out-of-sample predictions via walk-forward
    2. Computes calibration metrics per horizon
    3. Returns diagnostic results
    
    Args:
        px: Price series
        horizons: List of forecast horizons to test (default [1, 21, 63] = 1d, 1mo, 3mo)
        n_bins: Number of bins for reliability diagram (default 10)
        train_days: Training window size (default 504 = ~2 years)
        test_days: Test window size (default 21 = ~1 month)
        max_predictions: Maximum predictions to collect (default 500)
        
    Returns:
        Dictionary mapping horizon to CalibrationMetrics
    """
    # Collect predictions
    predictions = collect_predictions_walk_forward(
        px=px,
        horizons=horizons,
        train_days=train_days,
        test_days=test_days,
        max_predictions=max_predictions
    )
    
    # Compute calibration metrics per horizon
    calibration_results = {}
    
    for H in horizons:
        pred_probs = predictions[H]['predicted_probs']
        actual_rets = predictions[H]['actual_returns']
        
        if len(pred_probs) < 20:
            # Insufficient data for this horizon
            continue
        
        try:
            metrics = compute_pit_calibration(
                predicted_probs=pred_probs,
                actual_outcomes=actual_rets,
                n_bins=n_bins,
                ece_threshold=0.05
            )
            calibration_results[H] = metrics
        except Exception:
            # Skip horizons where calibration computation fails
            continue
    
    return calibration_results


def format_calibration_report(
    calibration_results: Dict[int, CalibrationMetrics],
    asset_name: str = "Asset"
) -> str:
    """
    Format calibration test results as human-readable report.
    
    Args:
        calibration_results: Dictionary mapping horizon to CalibrationMetrics
        asset_name: Name of asset for report title
        
    Returns:
        Formatted string report
    """
    from decision.signals_ux import format_horizon_label
    
    lines = []
    lines.append(f"═══════════════════════════════════════════════════════")
    lines.append(f"  PIT CALIBRATION REPORT: {asset_name}")
    lines.append(f"═══════════════════════════════════════════════════════")
    lines.append("")
    
    if not calibration_results:
        lines.append("⚠️  Insufficient data for calibration testing")
        return "\n".join(lines)
    
    for horizon, metrics in sorted(calibration_results.items()):
        lines.append(f"📊 Horizon: {format_horizon_label(horizon)}")
        lines.append(f"   Predictions: {metrics.n_predictions}")
        lines.append("")
        
        # Calibration status
        if metrics.calibrated:
            status_icon = "✅"
            status_text = "WELL-CALIBRATED"
        else:
            status_icon = "❌"
            if metrics.calibration_diagnosis == "overconfident":
                status_text = "OVERCONFIDENT (predictions too high)"
            elif metrics.calibration_diagnosis == "underconfident":
                status_text = "UNDERCONFIDENT (predictions too low)"
            else:
                status_text = "MISCALIBRATED"
        
        lines.append(f"   Status: {status_icon} {status_text}")
        lines.append("")
        
        # Key metrics
        lines.append(f"   Expected Calibration Error (ECE): {metrics.expected_calibration_error:.4f}")
        lines.append(f"   Maximum Calibration Error (MCE): {metrics.maximum_calibration_error:.4f}")
        lines.append(f"   Brier Score: {metrics.brier_score:.4f}")
        lines.append("")
        
        # Uniformity test
        if metrics.uniformity_pvalue >= 0.05:
            uniform_status = "✅ PASS"
        else:
            uniform_status = "⚠️  FAIL"
        lines.append(f"   Uniformity Test: {uniform_status} (p={metrics.uniformity_pvalue:.3f})")
        lines.append("")
        
        # Reliability diagram (text-based)
        lines.append("   Reliability Diagram:")
        lines.append("   ┌─────────────────────────────────────────┐")
        lines.append("   │ Predicted →   Actual Frequency         │")
        lines.append("   ├─────────────────────────────────────────┤")
        
        for i in range(metrics.n_bins):
            if metrics.bin_counts[i] > 0:
                pred = metrics.predicted_probs[i]
                actual = metrics.actual_frequencies[i]
                deviation = pred - actual
                
                # Format with deviation indicator
                if abs(deviation) < 0.03:
                    dev_icon = "✓"
                elif deviation > 0:
                    dev_icon = "↑"  # overconfident
                else:
                    dev_icon = "↓"  # underconfident
                
                lines.append(f"   │  {pred:5.2f}  →  {actual:5.2f}  (n={metrics.bin_counts[i]:3d}) {dev_icon} │")
        
        lines.append("   └─────────────────────────────────────────┘")
        lines.append("")
        lines.append("   Legend: ✓ = good fit, ↑ = overconfident, ↓ = underconfident")
        lines.append("")
        lines.append("─────────────────────────────────────────────────────")
        lines.append("")
    
    # Summary
    all_calibrated = all(m.calibrated for m in calibration_results.values())
    if all_calibrated:
        lines.append("🎯 OVERALL: All horizons are well-calibrated")
        lines.append("   → Probabilities are truth-telling")
        lines.append("   → Kelly sizing is safe to use")
    else:
        lines.append("⚠️  OVERALL: Some horizons are miscalibrated")
        lines.append("   → Consider probability recalibration (isotonic regression)")
        lines.append("   → Be cautious with Kelly sizing")
    
    lines.append("")
    lines.append("═══════════════════════════════════════════════════════")
    
    return "\n".join(lines)
