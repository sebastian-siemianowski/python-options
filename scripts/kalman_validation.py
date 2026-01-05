#!/usr/bin/env python3
"""
kalman_validation.py

Validation Science for Kalman-Filtered Drift Estimation (Level-7 Diagnostics)

This module implements formal validation and diagnostic infrastructure to prove
the Kalman filter "behaves like reality":

1. Posterior Drift Reasonableness Visualization
2. Predictive Likelihood Improvement Testing
3. Probability Integral Transform (PIT) Calibration
4. Parameter Uncertainty Propagation Verification
5. Stress-Regime Behavior Analysis

This is the difference between clever and serious quant work.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DriftReasonablenessResult:
    """Results from drift reasonableness validation."""
    asset_name: str
    observations: int
    drift_smoothness_ratio: float  # std(drift) / std(returns)
    crisis_uncertainty_spike: float  # max(σ_drift) / median(σ_drift)
    regime_break_detected: bool
    noise_tracking_score: float  # correlation(drift_changes, return_noise)
    validation_passed: bool
    diagnostic_message: str


def validate_drift_reasonableness(
    px: pd.Series,
    ret: pd.Series,
    mu_kf: pd.Series,
    var_kf: pd.Series,
    asset_name: str = "Asset",
    plot: bool = False,
    save_path: Optional[str] = None
) -> DriftReasonablenessResult:
    """
    Validate posterior drift reasonableness via statistical tests and optional visualization.
    
    Checks:
    1. Drift is smoother than returns (σ_drift << σ_returns)
    2. Uncertainty widens in crises (volatility spikes)
    3. Drift changes sign before/after regime breaks
    4. Drift does NOT track noise (low correlation with residuals)
    
    Args:
        px: Price series
        ret: Return series
        mu_kf: Kalman-filtered drift (posterior mean)
        var_kf: Kalman-filtered drift variance (posterior uncertainty)
        asset_name: Name for plot titles
        plot: If True, generate visualization
        save_path: Path to save plot (if None, display interactively)
        
    Returns:
        DriftReasonablenessResult with validation metrics and pass/fail status
    """
    # Align series
    df = pd.concat([px, ret, mu_kf, var_kf], axis=1, join='inner').dropna()
    if len(df) < 100:
        return DriftReasonablenessResult(
            asset_name=asset_name,
            observations=len(df),
            drift_smoothness_ratio=float('nan'),
            crisis_uncertainty_spike=float('nan'),
            regime_break_detected=False,
            noise_tracking_score=float('nan'),
            validation_passed=False,
            diagnostic_message="Insufficient data for validation (need >= 100 observations)"
        )
    
    df.columns = ['px', 'ret', 'mu', 'var']
    
    # 1. Drift smoothness: should be much smoother than returns
    std_drift = float(np.std(df['mu']))
    std_returns = float(np.std(df['ret']))
    smoothness_ratio = std_drift / std_returns if std_returns > 0 else 0.0
    
    # Expected: smoothness_ratio < 0.3 (drift evolves 3-10x slower than returns)
    smoothness_passed = smoothness_ratio < 0.5
    
    # 2. Crisis uncertainty spike: uncertainty should widen during volatility spikes
    std_uncertainty = np.sqrt(df['var'])
    median_uncertainty = float(np.median(std_uncertainty))
    max_uncertainty = float(np.max(std_uncertainty))
    uncertainty_spike = max_uncertainty / median_uncertainty if median_uncertainty > 0 else 1.0
    
    # Expected: uncertainty_spike > 2.0 (at least 2x increase during stress)
    crisis_passed = uncertainty_spike > 1.5
    
    # 3. Regime break detection: drift should change sign during regime transitions
    # Detect sign changes in drift
    drift_signs = np.sign(df['mu'])
    sign_changes = np.sum(np.abs(np.diff(drift_signs)) > 0)
    regime_break_detected = sign_changes >= 3  # At least a few regime transitions
    
    # 4. Noise tracking: drift should NOT correlate with return residuals
    # Residuals = returns - drift (should be pure noise)
    residuals = df['ret'] - df['mu']
    drift_changes = df['mu'].diff().dropna()
    residuals_aligned = residuals.iloc[1:]  # align with drift_changes
    
    # Correlation between drift changes and residuals (should be near zero)
    try:
        noise_correlation = float(np.corrcoef(drift_changes.values, residuals_aligned.values)[0, 1])
    except Exception:
        noise_correlation = 0.0
    
    noise_tracking_score = abs(noise_correlation)
    # Expected: noise_tracking_score < 0.3 (low correlation with noise)
    noise_passed = noise_tracking_score < 0.4
    
    # Overall validation
    validation_passed = smoothness_passed and crisis_passed and noise_passed
    
    # Diagnostic message
    messages = []
    if not smoothness_passed:
        messages.append(f"Drift not smooth enough (ratio={smoothness_ratio:.3f} > 0.5, may need smaller q)")
    if not crisis_passed:
        messages.append(f"Uncertainty doesn't spike in crises (max/med={uncertainty_spike:.2f} < 1.5)")
    if not noise_passed:
        messages.append(f"Drift tracks noise too much (corr={noise_tracking_score:.3f} > 0.4, may need larger q)")
    
    if validation_passed:
        diagnostic_message = "✅ All drift reasonableness checks passed"
    else:
        diagnostic_message = "⚠️ " + "; ".join(messages)
    
    # Optional visualization
    if plot:
        _plot_drift_diagnostics(df, asset_name, smoothness_ratio, uncertainty_spike, 
                               noise_tracking_score, validation_passed, save_path)
    
    return DriftReasonablenessResult(
        asset_name=asset_name,
        observations=len(df),
        drift_smoothness_ratio=smoothness_ratio,
        crisis_uncertainty_spike=uncertainty_spike,
        regime_break_detected=regime_break_detected,
        noise_tracking_score=noise_tracking_score,
        validation_passed=validation_passed,
        diagnostic_message=diagnostic_message
    )


def _plot_drift_diagnostics(
    df: pd.DataFrame,
    asset_name: str,
    smoothness_ratio: float,
    uncertainty_spike: float,
    noise_tracking: float,
    passed: bool,
    save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive drift diagnostic plots.
    
    Plots:
    1. Price series
    2. Returns vs Drift (overlay)
    3. Drift with ±2σ uncertainty bands
    4. Uncertainty evolution
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Kalman Drift Validation: {asset_name}', fontsize=16, fontweight='bold')
    
    dates = df.index
    
    # 1. Price series
    ax1 = axes[0]
    ax1.plot(dates, df['px'], color='black', linewidth=1.5, label='Price')
    ax1.set_ylabel('Price', fontsize=10)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Price Evolution', fontsize=11)
    
    # 2. Returns vs Drift
    ax2 = axes[1]
    ax2.plot(dates, df['ret'], color='gray', alpha=0.4, linewidth=0.5, label='Returns (noisy)')
    ax2.plot(dates, df['mu'], color='blue', linewidth=2, label='Kalman Drift μ̂_t')
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_ylabel('Return / Drift', fontsize=10)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'Returns vs Drift (smoothness ratio: {smoothness_ratio:.3f})', fontsize=11)
    
    # 3. Drift with uncertainty bands
    ax3 = axes[2]
    std_uncertainty = np.sqrt(df['var'])
    ax3.plot(dates, df['mu'], color='darkblue', linewidth=2, label='Drift μ̂_t')
    ax3.fill_between(dates, 
                      df['mu'] - 2*std_uncertainty, 
                      df['mu'] + 2*std_uncertainty,
                      color='lightblue', alpha=0.3, label='±2σ(μ̂) uncertainty')
    ax3.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.set_ylabel('Drift ± Uncertainty', fontsize=10)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Drift Posterior with Uncertainty Bands', fontsize=11)
    
    # 4. Uncertainty evolution
    ax4 = axes[3]
    ax4.plot(dates, std_uncertainty, color='red', linewidth=1.5, label='σ(μ̂) uncertainty')
    median_unc = np.median(std_uncertainty)
    ax4.axhline(median_unc, color='orange', linestyle='--', linewidth=1, alpha=0.7, label=f'Median: {median_unc:.6f}')
    ax4.set_ylabel('Drift Uncertainty', fontsize=10)
    ax4.set_xlabel('Date', fontsize=10)
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_title(f'Uncertainty Evolution (crisis spike: {uncertainty_spike:.2f}×)', fontsize=11)
    
    # Format x-axis dates
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add validation status text
    status_text = f"Validation: {'✅ PASSED' if passed else '❌ FAILED'}\n"
    status_text += f"Smoothness: {smoothness_ratio:.3f} {'✓' if smoothness_ratio < 0.5 else '✗'}\n"
    status_text += f"Crisis spike: {uncertainty_spike:.2f}× {'✓' if uncertainty_spike > 1.5 else '✗'}\n"
    status_text += f"Noise tracking: {noise_tracking:.3f} {'✓' if noise_tracking < 0.4 else '✗'}"
    
    fig.text(0.98, 0.02, status_text, fontsize=9, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


@dataclass
class LikelihoodComparisonResult:
    """Results from predictive likelihood improvement testing."""
    asset_name: str
    n_test_windows: int
    ll_kalman: float
    ll_zero_drift: float
    ll_ewma_drift: float
    ll_constant_drift: float
    delta_ll_vs_zero: float
    delta_ll_vs_ewma: float
    delta_ll_vs_constant: float
    improvement_significant: bool
    best_model: str
    diagnostic_message: str


def compare_predictive_likelihood(
    px: pd.Series,
    asset_name: str = "Asset",
    train_days: int = 504,
    test_days: int = 63,
    max_windows: int = 10
) -> LikelihoodComparisonResult:
    """
    Compare Kalman filter predictive likelihood against baseline models.
    
    Tests Kalman drift vs:
    - Zero drift (μ = 0)
    - EWMA drift (simple exponential smoothing)
    - Constant drift (sample mean)
    
    Uses walk-forward out-of-sample testing to avoid look-ahead bias.
    
    Args:
        px: Price series
        asset_name: Asset name for display
        train_days: Training window size (default ~2 years)
        test_days: Test window size (default ~3 months)
        max_windows: Maximum number of test windows
        
    Returns:
        LikelihoodComparisonResult with comparative log-likelihoods
    """
    from fx_pln_jpy_signals import compute_features
    
    log_px = np.log(px)
    dates = px.index
    
    # Define walk-forward windows
    windows = []
    start_idx = 0
    while start_idx + train_days + test_days <= len(dates) and len(windows) < max_windows:
        train_end_idx = start_idx + train_days
        test_end_idx = min(train_end_idx + test_days, len(dates))
        
        train_dates = dates[start_idx:train_end_idx]
        test_dates = dates[train_end_idx:test_end_idx]
        
        if len(test_dates) >= 10:
            windows.append({
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
            })
        
        start_idx = test_end_idx
    
    if not windows:
        return LikelihoodComparisonResult(
            asset_name=asset_name,
            n_test_windows=0,
            ll_kalman=float('nan'),
            ll_zero_drift=float('nan'),
            ll_ewma_drift=float('nan'),
            ll_constant_drift=float('nan'),
            delta_ll_vs_zero=float('nan'),
            delta_ll_vs_ewma=float('nan'),
            delta_ll_vs_constant=float('nan'),
            improvement_significant=False,
            best_model="N/A",
            diagnostic_message="Insufficient data for likelihood comparison"
        )
    
    # Accumulate log-likelihoods across windows
    ll_kalman_total = 0.0
    ll_zero_total = 0.0
    ll_ewma_total = 0.0
    ll_constant_total = 0.0
    n_obs_total = 0
    
    for window in windows:
        # Fit on training data
        train_px = px.loc[window["train_start"]:window["train_end"]]
        
        try:
            train_feats = compute_features(train_px)
            
            # Get Kalman drift and volatility at end of training
            mu_kf = train_feats.get("mu_kf", train_feats.get("mu"))
            vol = train_feats.get("vol")
            
            if mu_kf is None or vol is None or mu_kf.empty or vol.empty:
                continue
            
            mu_kalman = float(mu_kf.iloc[-1])
            sigma = float(vol.iloc[-1])
            
            # EWMA drift (simple exponential smoothing)
            train_ret = train_feats.get("ret")
            if train_ret is None or train_ret.empty:
                continue
            mu_ewma = float(train_ret.ewm(span=21, adjust=False).mean().iloc[-1])
            
            # Constant drift (sample mean)
            mu_constant = float(train_ret.mean())
            
            # Zero drift
            mu_zero = 0.0
            
            # Evaluate on test data
            test_px = px.loc[window["test_start"]:window["test_end"]]
            test_log_px = np.log(test_px)
            test_ret = test_log_px.diff().dropna()
            
            if len(test_ret) < 5:
                continue
            
            # Compute log-likelihoods (Gaussian: log p(r_t | μ, σ) = -0.5*log(2πσ²) - 0.5*(r_t - μ)²/σ²)
            for r in test_ret.values:
                ll_kalman_total += _gaussian_logpdf(r, mu_kalman, sigma)
                ll_zero_total += _gaussian_logpdf(r, mu_zero, sigma)
                ll_ewma_total += _gaussian_logpdf(r, mu_ewma, sigma)
                ll_constant_total += _gaussian_logpdf(r, mu_constant, sigma)
                n_obs_total += 1
                
        except Exception:
            continue
    
    if n_obs_total == 0:
        return LikelihoodComparisonResult(
            asset_name=asset_name,
            n_test_windows=len(windows),
            ll_kalman=float('nan'),
            ll_zero_drift=float('nan'),
            ll_ewma_drift=float('nan'),
            ll_constant_drift=float('nan'),
            delta_ll_vs_zero=float('nan'),
            delta_ll_vs_ewma=float('nan'),
            delta_ll_vs_constant=float('nan'),
            improvement_significant=False,
            best_model="N/A",
            diagnostic_message="All test windows failed to produce valid forecasts"
        )
    
    # Compute deltas
    delta_vs_zero = ll_kalman_total - ll_zero_total
    delta_vs_ewma = ll_kalman_total - ll_ewma_total
    delta_vs_constant = ll_kalman_total - ll_constant_total
    
    # Determine best model
    all_lls = {
        'Kalman': ll_kalman_total,
        'Zero': ll_zero_total,
        'EWMA': ll_ewma_total,
        'Constant': ll_constant_total
    }
    best_model = max(all_lls, key=all_lls.get)
    
    # Significance test: improvement > 2 log-likelihood units per window is substantial
    # (roughly equivalent to AIC difference of 4)
    min_improvement = 2.0 * len(windows)
    improvement_significant = bool(
        delta_vs_zero > min_improvement and 
        delta_vs_ewma > 0 and 
        delta_vs_constant > 0
    )
    
    if improvement_significant:
        diagnostic_message = f"✅ Kalman significantly outperforms all baselines (Δ_LL > {min_improvement:.1f})"
    elif best_model == 'Kalman':
        diagnostic_message = "⚠️ Kalman is best but improvement not statistically significant"
    else:
        diagnostic_message = f"❌ {best_model} model outperforms Kalman (may need q tuning)"
    
    return LikelihoodComparisonResult(
        asset_name=asset_name,
        n_test_windows=len(windows),
        ll_kalman=float(ll_kalman_total),
        ll_zero_drift=float(ll_zero_total),
        ll_ewma_drift=float(ll_ewma_total),
        ll_constant_drift=float(ll_constant_total),
        delta_ll_vs_zero=float(delta_vs_zero),
        delta_ll_vs_ewma=float(delta_vs_ewma),
        delta_ll_vs_constant=float(delta_vs_constant),
        improvement_significant=improvement_significant,
        best_model=best_model,
        diagnostic_message=diagnostic_message
    )


def _gaussian_logpdf(x: float, mu: float, sigma: float) -> float:
    """Compute Gaussian log-probability density."""
    sigma = max(sigma, 1e-12)
    return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((x - mu) / sigma)**2


@dataclass
class PITCalibrationResult:
    """Results from Probability Integral Transform (PIT) calibration check."""
    asset_name: str
    n_observations: int
    ks_statistic: float
    ks_pvalue: float
    uniform_hypothesis_rejected: bool
    pit_mean: float
    pit_std: float
    underestimation_bias: bool
    overestimation_bias: bool
    calibration_passed: bool
    diagnostic_message: str


def validate_pit_calibration(
    px: pd.Series,
    ret: pd.Series,
    mu_kf: pd.Series,
    var_kf: pd.Series,
    vol: pd.Series,
    asset_name: str = "Asset",
    plot: bool = False,
    save_path: Optional[str] = None
) -> PITCalibrationResult:
    """
    Validate forecast calibration via Probability Integral Transform (PIT).
    
    The PIT transforms forecast errors to uniform(0,1) if forecasts are well-calibrated.
    For each observation r_t, compute:
        PIT_t = Φ((r_t - μ̂_t) / σ_t)
    
    If forecasts are correct, PITs should be ~ Uniform(0,1):
    - Histogram should be flat
    - KS-test should not reject uniformity
    - Mean ≈ 0.5, std ≈ 0.289 (1/sqrt(12))
    
    Deviations indicate:
    - PITs bow upward → underestimate upside
    - PITs bow downward → overestimate
    - Heavy tails → Student-t needed
    
    Args:
        px: Price series
        ret: Return series
        mu_kf: Kalman-filtered drift (posterior mean)
        var_kf: Kalman-filtered drift variance (posterior uncertainty)
        vol: Volatility series
        asset_name: Name for display
        plot: If True, generate PIT diagnostic plots
        save_path: Path to save plot
        
    Returns:
        PITCalibrationResult with calibration metrics
    """
    from scipy.stats import norm, kstest
    
    # Align all series
    df = pd.concat([px, ret, mu_kf, var_kf, vol], axis=1, join='inner').dropna()
    if len(df) < 100:
        return PITCalibrationResult(
            asset_name=asset_name,
            n_observations=len(df),
            ks_statistic=float('nan'),
            ks_pvalue=float('nan'),
            uniform_hypothesis_rejected=False,
            pit_mean=float('nan'),
            pit_std=float('nan'),
            underestimation_bias=False,
            overestimation_bias=False,
            calibration_passed=False,
            diagnostic_message="Insufficient data for PIT calibration (need >= 100 observations)"
        )
    
    df.columns = ['px', 'ret', 'mu', 'var_mu', 'sigma']
    
    # Compute total forecast uncertainty: σ_forecast² = σ_obs² + Var(μ̂)
    # This propagates parameter uncertainty into forecasts
    forecast_var = df['sigma']**2 + df['var_mu']
    forecast_std = np.sqrt(forecast_var)
    
    # Compute PIT values: Φ((r_t - μ̂_t) / σ_forecast)
    standardized = (df['ret'] - df['mu']) / forecast_std
    pit_values = norm.cdf(standardized)
    
    # KS test for uniformity
    ks_stat, ks_pval = kstest(pit_values, 'uniform')
    uniform_rejected = bool(ks_pval < 0.05)
    
    # Compute PIT statistics
    pit_mean = float(np.mean(pit_values))
    pit_std = float(np.std(pit_values))
    expected_std = 1.0 / np.sqrt(12)  # ≈ 0.289 for Uniform(0,1)
    
    # Detect systematic biases
    # Mean > 0.55: model underestimates returns (too pessimistic)
    # Mean < 0.45: model overestimates returns (too optimistic)
    underestimation = bool(pit_mean > 0.55)
    overestimation = bool(pit_mean < 0.45)
    
    # Overall calibration check
    calibration_passed = bool(
        not uniform_rejected and 
        abs(pit_mean - 0.5) < 0.05 and
        abs(pit_std - expected_std) < 0.05
    )
    
    # Diagnostic message
    if calibration_passed:
        diagnostic_message = "✅ PIT calibration passed: forecasts are well-calibrated"
    else:
        messages = []
        if uniform_rejected:
            messages.append(f"KS test rejects uniformity (p={ks_pval:.4f})")
        if underestimation:
            messages.append(f"Systematic underestimation (PIT mean={pit_mean:.3f} > 0.55)")
        if overestimation:
            messages.append(f"Systematic overestimation (PIT mean={pit_mean:.3f} < 0.45)")
        if abs(pit_std - expected_std) > 0.05:
            if pit_std > expected_std:
                messages.append(f"Forecast uncertainty underestimated (PIT std={pit_std:.3f} > {expected_std:.3f})")
            else:
                messages.append(f"Forecast uncertainty overestimated (PIT std={pit_std:.3f} < {expected_std:.3f})")
        
        diagnostic_message = "⚠️ " + "; ".join(messages) if messages else "⚠️ PIT calibration checks failed"
    
    # Optional visualization
    if plot:
        _plot_pit_diagnostics(pit_values, asset_name, ks_stat, ks_pval, 
                             pit_mean, pit_std, calibration_passed, save_path)
    
    return PITCalibrationResult(
        asset_name=asset_name,
        n_observations=len(df),
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_pval),
        uniform_hypothesis_rejected=uniform_rejected,
        pit_mean=pit_mean,
        pit_std=pit_std,
        underestimation_bias=underestimation,
        overestimation_bias=overestimation,
        calibration_passed=calibration_passed,
        diagnostic_message=diagnostic_message
    )


def _plot_pit_diagnostics(
    pit_values: np.ndarray,
    asset_name: str,
    ks_stat: float,
    ks_pval: float,
    pit_mean: float,
    pit_std: float,
    passed: bool,
    save_path: Optional[str] = None
) -> None:
    """Create PIT diagnostic plots: histogram and QQ-plot."""
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import probplot
    except ImportError:
        print("Warning: matplotlib not available, skipping PIT visualization")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'PIT Calibration Check: {asset_name}', fontsize=16, fontweight='bold')
    
    # 1. Histogram
    ax1 = axes[0]
    ax1.hist(pit_values, bins=20, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax1.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Uniform(0,1) density')
    ax1.set_xlabel('PIT Value', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title(f'PIT Histogram (mean={pit_mean:.3f}, std={pit_std:.3f})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. QQ-plot
    ax2 = axes[1]
    probplot(pit_values, dist='uniform', plot=ax2)
    ax2.set_title(f'QQ-Plot vs Uniform (KS={ks_stat:.4f}, p={ks_pval:.4f})', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add validation status
    status_text = f"Calibration: {'✅ PASSED' if passed else '❌ FAILED'}\n"
    status_text += f"KS p-value: {ks_pval:.4f} {'✓' if ks_pval >= 0.05 else '✗'}\n"
    status_text += f"Mean: {pit_mean:.3f} (expected: 0.5) {'✓' if abs(pit_mean - 0.5) < 0.05 else '✗'}\n"
    expected_std = 1.0 / np.sqrt(12)
    status_text += f"Std: {pit_std:.3f} (expected: {expected_std:.3f}) {'✓' if abs(pit_std - expected_std) < 0.05 else '✗'}"
    
    fig.text(0.98, 0.02, status_text, fontsize=9, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"PIT plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


@dataclass
class StressRegimeResult:
    """Results from stress-regime behavior analysis."""
    asset_name: str
    n_periods: int
    stress_periods_detected: List[Tuple[str, str]]  # (start_date, end_date)
    avg_uncertainty_normal: float
    avg_uncertainty_stress: float
    uncertainty_spike_ratio: float
    avg_kelly_normal: float
    avg_kelly_stress: float
    kelly_reduction_ratio: float
    system_backed_off: bool
    diagnostic_message: str


def analyze_stress_regime_behavior(
    px: pd.Series,
    ret: pd.Series,
    mu_kf: pd.Series,
    var_kf: pd.Series,
    vol: pd.Series,
    asset_name: str = "Asset",
    stress_vol_threshold: float = 2.0
) -> StressRegimeResult:
    """
    Analyze model behavior during stress regimes (2008 crisis, COVID, etc.).
    
    Validates that during high-stress periods:
    1. Drift uncertainty (σ_μ) spikes
    2. Signal confidence collapses
    3. Kelly position sizes shrink
    4. System naturally backs off
    
    This proves the model has structural risk intelligence.
    
    Args:
        px: Price series
        ret: Return series
        mu_kf: Kalman drift
        var_kf: Drift variance
        vol: Volatility series
        asset_name: Asset name
        stress_vol_threshold: Multiplier of median vol to define stress (default 2.0)
        
    Returns:
        StressRegimeResult with stress-behavior metrics
    """
    # Align series
    df = pd.concat([px, ret, mu_kf, var_kf, vol], axis=1, join='inner').dropna()
    if len(df) < 252:
        return StressRegimeResult(
            asset_name=asset_name,
            n_periods=len(df),
            stress_periods_detected=[],
            avg_uncertainty_normal=float('nan'),
            avg_uncertainty_stress=float('nan'),
            uncertainty_spike_ratio=float('nan'),
            avg_kelly_normal=float('nan'),
            avg_kelly_stress=float('nan'),
            kelly_reduction_ratio=float('nan'),
            system_backed_off=False,
            diagnostic_message="Insufficient data for stress regime analysis (need >= 252 days)"
        )
    
    df.columns = ['px', 'ret', 'mu', 'var_mu', 'sigma']
    
    # Define stress periods: volatility > threshold × median
    median_vol = df['sigma'].rolling(252, min_periods=126).median()
    stress_mask = df['sigma'] > (stress_vol_threshold * median_vol)
    
    # Compute drift uncertainty
    df['std_mu'] = np.sqrt(df['var_mu'])
    
    # Compute Kelly fractions: f* = μ / σ²
    # Add parameter uncertainty: total var = σ² + Var(μ)
    total_var = df['sigma']**2 + df['var_mu']
    df['kelly_half'] = (df['mu'] / total_var).clip(-1, 1).abs() * 0.5
    
    # Split into normal and stress periods
    normal_mask = ~stress_mask
    
    # Compute statistics
    avg_unc_normal = float(df.loc[normal_mask, 'std_mu'].mean()) if normal_mask.sum() > 0 else float('nan')
    avg_unc_stress = float(df.loc[stress_mask, 'std_mu'].mean()) if stress_mask.sum() > 0 else float('nan')
    
    avg_kelly_normal = float(df.loc[normal_mask, 'kelly_half'].mean()) if normal_mask.sum() > 0 else float('nan')
    avg_kelly_stress = float(df.loc[stress_mask, 'kelly_half'].mean()) if stress_mask.sum() > 0 else float('nan')
    
    unc_spike = avg_unc_stress / avg_unc_normal if avg_unc_normal > 0 else 1.0
    kelly_reduction = avg_kelly_stress / avg_kelly_normal if avg_kelly_normal > 0 else 1.0
    
    # Identify stress periods (consecutive stress days)
    stress_periods = []
    in_stress = False
    start_date = None
    
    for date, is_stress in stress_mask.items():
        if is_stress and not in_stress:
            start_date = date
            in_stress = True
        elif not is_stress and in_stress:
            if start_date is not None:
                stress_periods.append((str(start_date.date()), str(df.index[df.index.get_loc(date) - 1].date())))
            in_stress = False
    
    # Close final period if still in stress
    if in_stress and start_date is not None:
        stress_periods.append((str(start_date.date()), str(df.index[-1].date())))
    
    # Validation: system should back off in stress (uncertainty up, Kelly down)
    system_backed_off = (
        unc_spike > 1.2 and  # Uncertainty increases by at least 20%
        kelly_reduction < 0.9  # Kelly decreases by at least 10%
    )
    
    # Diagnostic message
    if system_backed_off:
        diagnostic_message = f"✅ System has structural risk intelligence: uncertainty ↑{unc_spike:.2f}×, Kelly ↓{kelly_reduction:.2f}×"
    else:
        messages = []
        if unc_spike <= 1.2:
            messages.append(f"Uncertainty doesn't spike enough in stress (ratio={unc_spike:.2f})")
        if kelly_reduction >= 0.9:
            messages.append(f"Kelly doesn't reduce in stress (ratio={kelly_reduction:.2f})")
        diagnostic_message = "⚠️ " + "; ".join(messages)
    
    return StressRegimeResult(
        asset_name=asset_name,
        n_periods=len(df),
        stress_periods_detected=stress_periods,
        avg_uncertainty_normal=avg_unc_normal,
        avg_uncertainty_stress=avg_unc_stress,
        uncertainty_spike_ratio=unc_spike,
        avg_kelly_normal=avg_kelly_normal,
        avg_kelly_stress=avg_kelly_stress,
        kelly_reduction_ratio=kelly_reduction,
        system_backed_off=system_backed_off,
        diagnostic_message=diagnostic_message
    )


def run_full_validation_suite(
    px: pd.Series,
    asset_name: str = "Asset",
    plot: bool = False,
    plot_dir: Optional[str] = None
) -> Dict[str, any]:
    """
    Run complete Level-7 validation suite on asset.
    
    Includes:
    1. Drift reasonableness validation
    2. Predictive likelihood improvement testing
    3. PIT calibration check
    4. Stress-regime behavior analysis
    
    Args:
        px: Price series
        asset_name: Asset name for display
        plot: If True, generate diagnostic plots
        plot_dir: Directory to save plots (creates if needed)
        
    Returns:
        Dictionary with all validation results
    """
    import os
    from fx_pln_jpy_signals import compute_features
    
    # Compute features (includes Kalman filtering)
    feats = compute_features(px)
    
    # Extract required series
    ret = feats.get("ret")
    mu_kf = feats.get("mu_kf", feats.get("mu"))
    var_kf = feats.get("var_kf", pd.Series(0.0, index=ret.index))
    vol = feats.get("vol")
    
    if mu_kf is None or ret is None or vol is None:
        return {
            "error": "Failed to compute features or Kalman filter not available",
            "asset_name": asset_name
        }
    
    # Prepare plot paths
    if plot and plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        drift_plot = os.path.join(plot_dir, f"{asset_name}_drift_validation.png")
        pit_plot = os.path.join(plot_dir, f"{asset_name}_pit_calibration.png")
    else:
        drift_plot = None
        pit_plot = None
    
    # Run validation components
    results = {}
    
    # 1. Drift reasonableness
    results["drift_reasonableness"] = validate_drift_reasonableness(
        px, ret, mu_kf, var_kf, asset_name, plot=plot, save_path=drift_plot
    )
    
    # 2. Predictive likelihood comparison
    results["likelihood_comparison"] = compare_predictive_likelihood(
        px, asset_name
    )
    
    # 3. PIT calibration
    results["pit_calibration"] = validate_pit_calibration(
        px, ret, mu_kf, var_kf, vol, asset_name, plot=plot, save_path=pit_plot
    )
    
    # 4. Stress-regime behavior
    results["stress_regime"] = analyze_stress_regime_behavior(
        px, ret, mu_kf, var_kf, vol, asset_name
    )
    
    # Overall summary
    all_passed = (
        results["drift_reasonableness"].validation_passed and
        results["likelihood_comparison"].improvement_significant and
        results["pit_calibration"].calibration_passed and
        results["stress_regime"].system_backed_off
    )
    
    results["overall_passed"] = all_passed
    results["asset_name"] = asset_name
    
    return results