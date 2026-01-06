#!/usr/bin/env python3
"""
tune_q_mle.py

Automatic per-asset Kalman drift process-noise parameter (q) estimation via MLE.

Optimizes q by maximizing out-of-sample log-likelihood of returns under the
Gaussian state-space drift model, using EWMA volatility as observation variance.

Caches results persistently (JSON + CSV) for reuse across runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm, kstest

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from fx_data_utils import fetch_px, _download_prices, get_default_asset_universe


def load_asset_list(assets_arg: Optional[str], assets_file: Optional[str]) -> List[str]:
    """Load list of assets from command-line argument or file."""
    if assets_arg:
        return [a.strip() for a in assets_arg.split(',') if a.strip()]
    
    if assets_file and os.path.exists(assets_file):
        with open(assets_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Default asset list: use centralized universe from fx_data_utils
    return get_default_asset_universe()


def load_cache(cache_json: str) -> Dict[str, Dict]:
    """Load existing cache from JSON file."""
    if os.path.exists(cache_json):
        try:
            with open(cache_json, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return {}
    return {}


def save_cache(cache: Dict[str, Dict], cache_json: str, cache_csv: str) -> None:
    """Save cache to both JSON and CSV formats atomically."""
    # Create cache directory if needed
    os.makedirs(os.path.dirname(cache_json) if os.path.dirname(cache_json) else '.', exist_ok=True)
    
    # Write JSON (atomic via temp file) - stores full metadata
    json_temp = cache_json + '.tmp'
    with open(json_temp, 'w') as f:
        json.dump(cache, f, indent=2)
    os.replace(json_temp, cache_json)
    
    # Write CSV (human-friendly summary with key diagnostics)
    csv_rows = []
    for asset, data in cache.items():
        csv_rows.append({
            'asset': asset,
            'q': data.get('q'),
            'c': data.get('c', 1.0),  # Default to 1.0 for old cache entries
            'log_likelihood': data.get('log_likelihood'),
            'delta_ll_vs_zero': data.get('delta_ll_vs_zero', float('nan')),
            'ks_statistic': data.get('ks_statistic', float('nan')),
            'pit_ks_pvalue': data.get('pit_ks_pvalue'),
            'calibration_warning': data.get('calibration_warning', False),
            'mean_drift_var': data.get('mean_drift_var', float('nan')),
            'mean_posterior_unc': data.get('mean_posterior_unc', float('nan')),
            'n_obs': data.get('n_obs'),
            'fallback_reason': data.get('fallback_reason', ''),
            'timestamp': data.get('timestamp')
        })
    
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        csv_temp = cache_csv + '.tmp'
        df.to_csv(csv_temp, index=False)
        os.replace(csv_temp, cache_csv)


def kalman_filter_drift(returns: np.ndarray, vol: np.ndarray, q: float, c: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run Kalman filter for drift estimation with fixed process noise q and observation variance scale c.
    
    State-space model:
        μ_t = μ_{t-1} + w_t,  w_t ~ N(0, q)         (state evolution)
        r_t = μ_t + v_t,      v_t ~ N(0, c·σ_t²)    (observation with scaled variance)
    
    Args:
        returns: Observed returns
        vol: EWMA volatility estimates
        q: Process noise variance (drift evolution)
        c: Observation variance scale factor (corrects EWMA bias)
    
    Returns:
        mu_filtered: Posterior mean of drift at each time step
        P_filtered: Posterior variance of drift at each time step
        log_likelihood: Total log-likelihood of observations
    """
    n = len(returns)
    
    # Ensure q and c are scalars
    q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, 'item') else float(q)
    c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, 'item') else float(c)
    
    # Initialize state
    mu = 0.0  # Initial drift estimate
    P = 1e-4  # Initial uncertainty
    
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    log_likelihood = 0.0
    
    for t in range(n):
        # Predict
        mu_pred = float(mu)
        P_pred = float(P) + q_val
        
        # Observation variance with scale factor (extract scalar from array)
        vol_t = vol[t]
        vol_scalar = float(vol_t) if np.ndim(vol_t) == 0 else float(vol_t.item())
        R = c_val * (vol_scalar ** 2)
        
        # Update (Kalman gain)
        K = P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0
        
        # Innovation (extract scalar from array)
        ret_t = returns[t]
        r_val = float(ret_t) if np.ndim(ret_t) == 0 else float(ret_t.item())
        innovation = r_val - mu_pred
        
        # Update state (keep as Python float)
        mu = float(mu_pred + K * innovation)
        P = float((1.0 - K) * P_pred)
        
        # Store filtered estimates
        mu_filtered[t] = mu
        P_filtered[t] = P
        
        # Accumulate log-likelihood: log p(r_t | past) = log N(r_t; μ_pred, P_pred + R)
        forecast_var = P_pred + R
        if forecast_var > 1e-12:
            log_likelihood += -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var
    
    return mu_filtered, P_filtered, log_likelihood


def compute_pit_ks_pvalue(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float = 1.0) -> Tuple[float, float]:
    """
    Compute PIT (Probability Integral Transform) and KS test statistic + p-value.
    
    Well-calibrated forecasts should have PIT ~ Uniform(0,1).
    
    Args:
        returns: Observed returns
        mu_filtered: Kalman filtered drift estimates
        vol: Volatility estimates
        P_filtered: Drift posterior variance
        c: Observation variance scale factor
    
    Returns:
        ks_statistic: KS test statistic
        ks_pvalue: KS test p-value
    """
    # Ensure all inputs are 1D arrays
    returns_flat = np.asarray(returns).flatten()
    mu_flat = np.asarray(mu_filtered).flatten()
    vol_flat = np.asarray(vol).flatten()
    P_flat = np.asarray(P_filtered).flatten()
    
    # Total forecast variance = scaled observation variance + parameter uncertainty
    forecast_std = np.sqrt(c * (vol_flat ** 2) + P_flat)
    
    # Standardize returns
    standardized = (returns_flat - mu_flat) / forecast_std
    
    # Compute PIT values
    pit_values = norm.cdf(standardized)
    
    # KS test against uniform distribution
    ks_result = kstest(pit_values, 'uniform')
    
    # Extract statistic and p-value
    return float(ks_result.statistic), float(ks_result.pvalue)


def optimize_q_mle(
    returns: np.ndarray,
    vol: np.ndarray,
    train_frac: float = 0.7,
    q_min: float = 1e-10,
    q_max: float = 1e-2,
    c_min: float = 0.5,
    c_max: float = 2.0,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0
) -> Tuple[float, float, float, Dict]:
    """
    Jointly optimize (q, c) via maximum likelihood with Bayesian regularization.
    
    Uses walk-forward validation:
    - Train on first train_frac of data to warm up Kalman filter
    - Evaluate log-likelihood on remaining data
    - Apply prior regularization: log p(q) = -λ * (log_q - μ)²
    
    Args:
        returns: Return series
        vol: Volatility series
        train_frac: Fraction of data for training
        q_min, q_max: Bounds for process noise q
        c_min, c_max: Bounds for observation variance scale c
        prior_log_q_mean: Prior mean for log10(q) (default: -6)
        prior_lambda: Regularization strength (default: 1.0)
    
    Returns:
        q_optimal: Best-fit process noise
        c_optimal: Best-fit observation variance scale
        ll_optimal: Out-of-sample log-likelihood at optimum
        diagnostics: Dictionary with optimization diagnostics
    """
    n = len(returns)
    split_idx = int(n * train_frac)
    
    if split_idx < 50 or (n - split_idx) < 20:
        # Insufficient data for proper train/test split
        split_idx = max(50, n - 20)
    
    def negative_penalized_ll(params: np.ndarray) -> float:
        """Objective: negative penalized out-of-sample log-likelihood."""
        log_q, log_c = params
        q = 10 ** log_q
        c = 10 ** log_c
        
        # Run Kalman filter on full data with (q, c)
        mu_filt, P_filt, _ = kalman_filter_drift(returns, vol, q, c)
        
        # Compute out-of-sample log-likelihood (test set only)
        ll_oos = 0.0
        for t in range(split_idx, n):
            # Use previous state as prediction
            if t > 0:
                mu_pred = mu_filt[t-1]
                P_pred = P_filt[t-1] + q
            else:
                mu_pred = 0.0
                P_pred = 1e-4
            
            R = c * (vol[t] ** 2)
            innovation = returns[t] - mu_pred
            forecast_var = P_pred + R
            
            if forecast_var > 1e-12:
                ll_oos += -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var
        
        # Add Bayesian prior regularization on q
        log_prior = -prior_lambda * (log_q - prior_log_q_mean) ** 2
        
        # Total penalized likelihood
        penalized_ll = ll_oos + log_prior
        
        return -penalized_ll  # Minimize negative = maximize
    
    # Grid search in log-space for global optimum (2D)
    log_q_min = np.log10(q_min)
    log_q_max = np.log10(q_max)
    log_c_min = np.log10(c_min)
    log_c_max = np.log10(c_max)
    
    # Coarse 2D grid search (10×10 = 100 evaluations)
    log_q_grid = np.linspace(log_q_min, log_q_max, 10)
    log_c_grid = np.linspace(log_c_min, log_c_max, 10)
    
    best_neg_ll = float('inf')
    best_log_q_grid = log_q_grid[len(log_q_grid)//2]  # Middle of range
    best_log_c_grid = 0.0  # c=1.0
    
    for lq in log_q_grid:
        for lc in log_c_grid:
            try:
                neg_ll = negative_penalized_ll(np.array([lq, lc]))
                if neg_ll < best_neg_ll:
                    best_neg_ll = neg_ll
                    best_log_q_grid = lq
                    best_log_c_grid = lc
            except Exception:
                continue
    
    # Store grid search result for diagnostics
    grid_best_q = 10 ** best_log_q_grid
    grid_best_c = 10 ** best_log_c_grid
    
    # Fine optimization via bounded minimize
    bounds = [(log_q_min, log_q_max), (log_c_min, log_c_max)]
    x0 = np.array([best_log_q_grid, best_log_c_grid])
    
    try:
        result = minimize(
            negative_penalized_ll,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        if result.success:
            log_q_opt, log_c_opt = result.x
            q_optimal = 10 ** log_q_opt
            c_optimal = 10 ** log_c_opt
            ll_optimal = -result.fun
        else:
            # Fallback to grid search
            q_optimal = grid_best_q
            c_optimal = grid_best_c
            ll_optimal = -best_neg_ll
    except Exception:
        # Fallback to grid search
        q_optimal = grid_best_q
        c_optimal = grid_best_c
        ll_optimal = -best_neg_ll
    
    # Diagnostics
    diagnostics = {
        'grid_best_q': float(grid_best_q),
        'grid_best_c': float(grid_best_c),
        'refined_best_q': float(q_optimal),
        'refined_best_c': float(c_optimal),
        'prior_applied': prior_lambda > 0,
        'prior_log_q_mean': float(prior_log_q_mean),
        'prior_lambda': float(prior_lambda)
    }
    
    return q_optimal, c_optimal, ll_optimal, diagnostics


def tune_asset_q(
    asset: str,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0
) -> Optional[Dict]:
    """
    Estimate optimal (q, c) for a single asset via joint MLE with safety checks.
    
    Includes:
    - Joint (q, c) optimization with Bayesian regularization
    - Zero-drift baseline comparison (ΔLL)
    - Safety fallbacks (q collapse, miscalibration, worse than baseline)
    - Comprehensive diagnostic metadata
    
    Args:
        asset: Asset symbol
        start_date: Start date for data
        end_date: End date (default: today)
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
    
    Returns:
        Dictionary with results and diagnostics, or None if estimation failed
    """
    try:
        # Fetch price data
        try:
            px, title = fetch_px(asset, start_date, end_date)
        except Exception:
            # Fallback: direct download
            df = _download_prices(asset, start_date, end_date)
            if df.empty:
                return None
            px = df['Close']
            title = asset
        
        if len(px) < 252:
            print(f"  ⚠️  {asset}: Insufficient data ({len(px)} days)")
            return None
        
        # Compute returns
        log_px = np.log(px)
        returns = log_px.diff().dropna()
        
        # Compute EWMA volatility (observation noise)
        vol = returns.ewm(span=21, adjust=False).std()
        
        # Align series
        returns = returns.iloc[20:]  # Skip initial EWMA warmup
        vol = vol.iloc[20:]
        
        returns_arr = returns.values
        vol_arr = vol.values
        
        # Optimize (q, c) via joint MLE with regularization
        q_optimal, c_optimal, ll_optimal, opt_diagnostics = optimize_q_mle(
            returns_arr, vol_arr,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )
        
        # Run full Kalman filter with optimal (q, c)
        mu_filtered, P_filtered, ll_full = kalman_filter_drift(returns_arr, vol_arr, q_optimal, c_optimal)
        
        # Compute zero-drift baseline for comparison
        ll_zero = 0.0
        for t in range(len(returns_arr)):
            # Extract scalar values
            ret_t = float(returns_arr[t]) if np.ndim(returns_arr[t]) == 0 else float(returns_arr[t].item())
            vol_t = float(vol_arr[t]) if np.ndim(vol_arr[t]) == 0 else float(vol_arr[t].item())
            
            R = c_optimal * (vol_t ** 2)
            innovation = ret_t - 0.0
            forecast_var = R
            if forecast_var > 1e-12:
                ll_zero += -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var
        
        # Ensure scalar
        ll_zero = float(ll_zero)
        delta_ll_vs_zero = float(ll_full - ll_zero)
        
        # Compute PIT KS statistic and p-value for calibration check
        ks_statistic, ks_pvalue = compute_pit_ks_pvalue(returns_arr, mu_filtered, vol_arr, P_filtered, c_optimal)
        
        # Compute drift diagnostics
        mean_drift_var = float(np.mean(mu_filtered ** 2))
        mean_posterior_unc = float(np.mean(P_filtered))
        
        # Safety checks and fallback logic
        fallback_reason = None
        calibration_warning = False
        
        # Check 1: q collapse (too small)
        if q_optimal < 1e-10:
            fallback_reason = "q_collapsed_below_1e-10"
            q_optimal = 1e-6  # Heuristic fallback
            print(f"  ⚠️  q collapsed to {q_optimal:.2e}, using fallback q=1e-6")
        
        # Check 2: Severe miscalibration (PIT p < 0.01)
        if ks_pvalue < 0.01:
            fallback_reason = fallback_reason or "severe_miscalibration_pit_p<0.01"
            print(f"  ⚠️  Severe miscalibration (PIT p={ks_pvalue:.4f} < 0.01)")
        
        # Check 3: Miscalibration warning (PIT p < 0.05)
        if ks_pvalue < 0.05:
            calibration_warning = True
            print(f"  ⚠️  Calibration warning (PIT p={ks_pvalue:.4f} < 0.05)")
        
        # Check 4: Worse than zero-drift baseline
        if delta_ll_vs_zero < 0:
            fallback_reason = fallback_reason or "worse_than_zero_drift_baseline"
            print(f"  ⚠️  LL worse than zero-drift baseline (ΔLL={delta_ll_vs_zero:.2f})")
        
        # Build result dictionary with comprehensive metadata
        result = {
            'asset': asset,
            'q': float(q_optimal),
            'c': float(c_optimal),
            'log_likelihood': float(ll_full),
            'delta_ll_vs_zero': float(delta_ll_vs_zero),
            'ks_statistic': float(ks_statistic),
            'pit_ks_pvalue': float(ks_pvalue),
            'calibration_warning': bool(calibration_warning),
            'mean_drift_var': float(mean_drift_var),
            'mean_posterior_unc': float(mean_posterior_unc),
            'n_obs': int(len(returns_arr)),
            'timestamp': datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            'fallback_reason': fallback_reason,
            # Optimization diagnostics
            'grid_best_q': opt_diagnostics['grid_best_q'],
            'grid_best_c': opt_diagnostics['grid_best_c'],
            'refined_best_q': opt_diagnostics['refined_best_q'],
            'refined_best_c': opt_diagnostics['refined_best_c'],
            'prior_applied': opt_diagnostics['prior_applied'],
            'prior_log_q_mean': opt_diagnostics['prior_log_q_mean'],
            'prior_lambda': opt_diagnostics['prior_lambda']
        }
        
        return result
        
    except Exception as e:
        import traceback
        print(f"  ❌ {asset}: Failed - {e}")
        if os.getenv('DEBUG'):
            traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Estimate optimal Kalman drift parameters (q, c) via joint MLE with Bayesian regularization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --force                          # Re-estimate all assets
  %(prog)s --max-assets 10 --dry-run        # Preview first 10 assets
  %(prog)s --prior-lambda 2.0 --prior-mean -5.5  # Custom regularization
  %(prog)s --debug                          # Enable debug output
        """
    )
    parser.add_argument('--assets', type=str, help='Comma-separated list of asset symbols')
    parser.add_argument('--assets-file', type=str, help='Path to file with asset list (one per line)')
    parser.add_argument('--cache-json', type=str, default='cache/kalman_q_cache.json',
                       help='Path to JSON cache file')
    parser.add_argument('--cache-csv', type=str, default='cache/kalman_q_cache.csv',
                       help='Path to CSV cache file')
    parser.add_argument('--force', action='store_true',
                       help='Force re-estimation even if cached values exist')
    parser.add_argument('--start', type=str, default='2015-01-01',
                       help='Start date for data fetching')
    parser.add_argument('--end', type=str, default=None,
                       help='End date for data fetching (default: today)')
    
    # CLI enhancements
    parser.add_argument('--max-assets', type=int, default=None,
                       help='Maximum number of assets to process (useful for testing)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what would be done without actually processing')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output (stack traces on errors)')
    
    # Bayesian regularization parameters
    parser.add_argument('--prior-mean', type=float, default=-6.0,
                       help='Prior mean for log10(q) (default: -6.0)')
    parser.add_argument('--prior-lambda', type=float, default=1.0,
                       help='Regularization strength (default: 1.0, set to 0 to disable)')
    
    args = parser.parse_args()
    
    # Enable debug mode
    if args.debug:
        os.environ['DEBUG'] = '1'
    
    print("=" * 80)
    print("Kalman (q, c) Joint MLE Tuning Pipeline")
    print("=" * 80)
    print(f"Prior: log10(q) ~ N({args.prior_mean:.1f}, λ={args.prior_lambda:.1f})")
    
    # Load asset list
    assets = load_asset_list(args.assets, args.assets_file)
    
    # Apply max-assets limit
    if args.max_assets:
        assets = assets[:args.max_assets]
        print(f"\nLimited to first {args.max_assets} assets")
    
    print(f"Assets to process: {len(assets)}")
    
    # Dry-run mode
    if args.dry_run:
        print("\n[DRY RUN MODE - No actual processing]")
        print("Would process:")
        for i, asset in enumerate(assets[:10], 1):
            print(f"  {i}. {asset}")
        if len(assets) > 10:
            print(f"  ... and {len(assets) - 10} more")
        return
    
    # Load existing cache
    cache = load_cache(args.cache_json)
    print(f"Loaded cache with {len(cache)} existing entries")
    
    # Process each asset
    new_estimates = 0
    reused_cached = 0
    failed = 0
    calibration_warnings = 0
    
    for i, asset in enumerate(assets, 1):
        print(f"\n[{i}/{len(assets)}] {asset}")
        
        # Check cache
        if not args.force and asset in cache:
            cached_q = cache[asset].get('q', float('nan'))
            cached_c = cache[asset].get('c', 1.0)
            print(f"  ✓ Using cached estimate (q={cached_q:.2e}, c={cached_c:.3f})")
            reused_cached += 1
            continue
        
        # Estimate (q, c)
        result = tune_asset_q(
            asset, 
            args.start, 
            args.end,
            prior_log_q_mean=args.prior_mean,
            prior_lambda=args.prior_lambda
        )
        
        if result:
            cache[asset] = result
            new_estimates += 1
            
            # Count calibration warnings
            if result.get('calibration_warning'):
                calibration_warnings += 1
            
            # Display result
            q_val = result['q']
            c_val = result['c']
            ll_val = result['log_likelihood']
            delta_ll = result.get('delta_ll_vs_zero', float('nan'))
            pit_p = result['pit_ks_pvalue']
            
            print(f"  ✓ q={q_val:.2e}, c={c_val:.3f}, ΔLL={delta_ll:+.1f}, PIT p={pit_p:.3f}")
        else:
            failed += 1
    
    # Save updated cache
    if new_estimates > 0:
        save_cache(cache, args.cache_json, args.cache_csv)
        print(f"\n✓ Cache updated: {args.cache_json}, {args.cache_csv}")
    
    # Summary report
    print("\n" + "=" * 80)
    print("Kalman (q, c) Joint MLE Tuning Summary")
    print("=" * 80)
    print(f"Assets processed:       {len(assets)}")
    print(f"New estimates:          {new_estimates}")
    print(f"Reused cached:          {reused_cached}")
    print(f"Failed:                 {failed}")
    print(f"Calibration warnings:   {calibration_warnings}")
    
    if cache:
        print(f"\nBest-fit parameters (sorted by q):")
        print(f"{'Asset':<20} {'log10(q)':<10} {'c':<8} {'ΔLL':<8} {'PIT p':<8}")
        print("-" * 62)
        
        # Sort by q value for display
        sorted_assets = sorted(cache.items(), key=lambda x: x[1].get('q', 0), reverse=True)
        for asset, data in sorted_assets[:20]:  # Show top 20
            q_val = data.get('q', float('nan'))
            c_val = data.get('c', 1.0)
            delta_ll = data.get('delta_ll_vs_zero', float('nan'))
            pit_p = data.get('pit_ks_pvalue', float('nan'))
            
            log10_q = np.log10(q_val) if q_val > 0 else float('nan')
            
            # Mark calibration warnings
            pit_str = f"{pit_p:.3f}"
            if data.get('calibration_warning'):
                pit_str += " ⚠️"
            
            print(f"{asset:<20} {log10_q:>8.2f}   {c_val:>6.3f}  {delta_ll:>6.1f}  {pit_str:<8}")
        
        if len(cache) > 20:
            print(f"... and {len(cache) - 20} more")
        
        print(f"\nCache files:")
        print(f"  JSON: {args.cache_json}")
        print(f"  CSV:  {args.cache_csv}")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
