#!/usr/bin/env python3
"""
Test Gaussian Model PIT Calibration (including momentum flavor).

This test runs PIT calibration tests for Gaussian-based models on the same
asset set as the unified Student-t test (make pit).

Models tested:
    - φ-Gaussian: AR(1) drift with Gaussian noise
    - φ-Gaussian+Momentum: AR(1) drift with momentum augmentation
    - Gaussian: Pure Kalman with no AR(1) drift (baseline)

Usage:
    make pit-g           # Run all Gaussian PIT tests
    python test_gaussian_pit_failures.py --quick   # Quick test (5 assets)
    python test_gaussian_pit_failures.py --full    # Full test (22 assets)
"""

import sys
import os
import warnings
import argparse

warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

print('Loading Gaussian PIT test module...')

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, asdict
from scipy.stats import kstest, norm

# Same assets as make pit (unified Student-t test)
TEST_ASSETS = [
    'CDNS', 'CRM', 'ADBE', 'ADI', 'ISRG', 'SAIC', 'FDX', 'BKNG',
    'LLY', 'AIG', 'DE', 'HII', 'HEI', 'MET', 'JCI', 'GOOGL',
    'TMO', 'GOOG', 'UNH', 'USB', 'BAH', 'LDOS',
]

QUICK_TEST_ASSETS = ['GOOGL', 'CDNS', 'CRM', 'ISRG', 'LLY']

PIT_PVALUE_THRESHOLD = 0.05
MAD_THRESHOLD = 0.05


@dataclass
class GaussianPITResult:
    """Result from Gaussian PIT calibration test."""
    symbol: str
    model_type: str  # 'phi_gaussian', 'phi_gaussian_momentum', 'gaussian'
    pit_pvalue: float
    ks_statistic: float
    histogram_mad: float
    calibration_grade: str
    log10_q: float
    c: float
    phi: float
    nu: float  # Degrees of freedom (0 for Gaussian)
    alpha: float  # Asymmetry parameter (0 for Gaussian)
    gamma: float  # VoV or persistence (0 for Gaussian)
    log_likelihood: float
    bic: float
    hyvarinen: float  # Hyvarinen score
    crps: float  # CRPS score
    n_obs: int
    fit_success: bool
    error: Optional[str] = None
    
    @property
    def pit_failed(self) -> bool:
        return self.pit_pvalue < PIT_PVALUE_THRESHOLD
    
    @property
    def mad_failed(self) -> bool:
        return self.histogram_mad > MAD_THRESHOLD


def fetch_asset_data(symbol: str, start_date: str = '2015-01-01') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Fetch price data and compute returns + volatility."""
    try:
        from tuning.tune import _download_prices, compute_hybrid_volatility_har
        
        df = _download_prices(symbol, start_date, None)  # end=None for latest
        if df is None or df.empty:
            return None, None
        
        cols = {c.lower(): c for c in df.columns}
        if 'close' not in cols:
            return None, None
        
        px = df[cols['close']]
        if len(px) < 100:
            return None, None
        
        log_ret = np.log(px / px.shift(1)).dropna()
        returns = log_ret.values
        
        # Compute volatility using hybrid HAR
        if all(c in cols for c in ['open', 'high', 'low', 'close']):
            df_aligned = df.iloc[1:].copy()
            vol, _ = compute_hybrid_volatility_har(
                open_=df_aligned[cols['open']].values,
                high=df_aligned[cols['high']].values,
                low=df_aligned[cols['low']].values,
                close=df_aligned[cols['close']].values,
                span=21, annualize=False, use_har=True
            )
        else:
            vol = log_ret.ewm(span=21).std().values
        
        min_len = min(len(returns), len(vol))
        returns = returns[:min_len]
        vol = vol[:min_len]
        
        valid_mask = np.isfinite(returns) & np.isfinite(vol) & (vol > 0)
        returns = returns[valid_mask]
        vol = vol[valid_mask]
        
        if len(returns) < 100:
            return None, None
        
        return returns, vol
    except Exception as e:
        print(f"Warning: Data fetch error for {symbol}: {e}")
        return None, None


def compute_bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """Compute BIC = -2*LL + k*ln(n)."""
    if not np.isfinite(log_likelihood) or n_obs <= 0:
        return float('inf')
    return -2 * log_likelihood + n_params * np.log(n_obs)


def compute_crps_gaussian(returns: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Compute CRPS for Gaussian predictive distribution.
    
    CRPS(F, y) = σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]
    where z = (y - μ) / σ
    """
    z = (returns - mu) / np.maximum(sigma, 1e-10)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    crps_values = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps_values))


def compute_hyvarinen_gaussian(returns: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Compute Hyvarinen score for Gaussian distribution.
    
    The Hyvarinen score is a proper scoring rule based on the score function.
    For Gaussian: H = (1/σ²) * [(y-μ)² / σ² - 1]
    
    Returns the mean Hyvarinen score (lower is better, 0 is perfect).
    """
    var = np.maximum(sigma ** 2, 1e-12)
    z_squared = ((returns - mu) ** 2) / var
    hyv_values = (1.0 / var) * (z_squared - 1.0)
    return float(np.mean(hyv_values))


def run_phi_gaussian_pit(
    symbol: str,
    returns: np.ndarray,
    vol: np.ndarray,
    with_momentum: bool = False,
) -> GaussianPITResult:
    """Run φ-Gaussian filter and compute PIT calibration."""
    from models.phi_gaussian import PhiGaussianDriftModel
    
    model_type = 'phi_gaussian_momentum' if with_momentum else 'phi_gaussian'
    n_obs = len(returns)
    
    try:
        # Optimize parameters - returns (q, c, phi, ll, diagnostics)
        q_opt, c_opt, phi_opt, ll_opt, diagnostics = PhiGaussianDriftModel.optimize_params(
            returns, vol
        )
        
        if not diagnostics.get('optimization_successful', True):
            return GaussianPITResult(
                symbol=symbol,
                model_type=model_type,
                pit_pvalue=0.0,
                ks_statistic=1.0,
                histogram_mad=1.0,
                calibration_grade='F',
                log10_q=float('nan'),
                c=float('nan'),
                phi=float('nan'),
                nu=0.0,
                alpha=0.0,
                gamma=0.0,
                log_likelihood=float('nan'),
                bic=float('inf'),
                hyvarinen=float(hyv),
                crps=float(crps),
                n_obs=n_obs,
                fit_success=False,
                error=diagnostics.get('error', 'Optimization failed')
            )
        
        # Run filter - use momentum wrapper if requested
        if with_momentum:
            from models.momentum_augmented import MomentumAugmentedDriftModel, MomentumConfig
            mom_config = MomentumConfig(enable=True, lookbacks=[5, 10, 20, 60])
            mom_model = MomentumAugmentedDriftModel(mom_config)
            mom_model.precompute_momentum(returns)
            mu_filt, P_filt, ll = mom_model.filter(
                returns, vol, q_opt, c_opt, phi_opt, base_model='phi_gaussian'
            )
        else:
            mu_filt, P_filt, ll = PhiGaussianDriftModel.filter(
                returns, vol, q_opt, c_opt, phi_opt
            )
        
        # Compute PIT values
        pit_values = []
        for t in range(len(returns)):
            if t == 0:
                mu_pred = 0.0
                P_pred = 1e-4 + q_opt
            else:
                mu_pred = phi_opt * mu_filt[t-1]
                P_pred = (phi_opt ** 2) * P_filt[t-1] + q_opt
            
            R = c_opt * (vol[t] ** 2)
            S = P_pred + R
            
            innovation = returns[t] - mu_pred
            z = innovation / np.sqrt(max(S, 1e-12))
            
            pit_t = norm.cdf(z)
            pit_values.append(pit_t)
        
        pit_values = np.clip(pit_values, 0.001, 0.999)
        
        # KS test
        ks_result = kstest(pit_values, 'uniform')
        ks_stat = float(ks_result.statistic)
        ks_pvalue = float(ks_result.pvalue)
        
        # Histogram MAD
        hist, _ = np.histogram(pit_values, bins=10, range=(0, 1))
        hist_freq = hist / len(pit_values)
        hist_mad = float(np.mean(np.abs(hist_freq - 0.1)))
        
        # Grade
        if hist_mad < 0.02:
            grade = "A"
        elif hist_mad < 0.05:
            grade = "B"
        elif hist_mad < 0.10:
            grade = "C"
        else:
            grade = "F"
        
        # BIC: 3 parameters (q, c, phi)
        bic = compute_bic(ll, 3, n_obs)
        
        # Compute CRPS and Hyvarinen scores
        forecast_std = np.sqrt(c_opt) * vol
        crps = compute_crps_gaussian(returns, mu_filt, forecast_std)
        hyv = compute_hyvarinen_gaussian(returns, mu_filt, forecast_std)
        
        return GaussianPITResult(
            symbol=symbol,
            model_type=model_type,
            pit_pvalue=ks_pvalue,
            ks_statistic=ks_stat,
            histogram_mad=hist_mad,
            calibration_grade=grade,
            log10_q=float(np.log10(q_opt)) if q_opt > 0 else float('-inf'),
            c=float(c_opt),
            phi=float(phi_opt),
            nu=0.0,
            alpha=0.0,
            gamma=0.0,
            log_likelihood=float(ll),
            bic=float(bic),
            hyvarinen=float(hyv),
            crps=float(crps),
            n_obs=n_obs,
            fit_success=True,
        )
    
    except Exception as e:
        return GaussianPITResult(
            symbol=symbol,
            model_type=model_type,
            pit_pvalue=0.0,
            ks_statistic=1.0,
            histogram_mad=1.0,
            calibration_grade='F',
            log10_q=float('nan'),
            c=float('nan'),
            phi=float('nan'),
                nu=0.0,
                alpha=0.0,
                gamma=0.0,
            log_likelihood=float('nan'),
            bic=float('inf'),
                hyvarinen=float(hyv),
                crps=float(crps),
            n_obs=n_obs,
            fit_success=False,
            error=str(e)
        )


def run_gaussian_pit(symbol: str, returns: np.ndarray, vol: np.ndarray) -> GaussianPITResult:
    """Run pure Gaussian filter (no AR(1) drift) and compute PIT calibration."""
    from models.gaussian import GaussianDriftModel
    
    n_obs = len(returns)
    
    try:
        # Optimize parameters - returns (q, c, ll, diagnostics)
        q_opt, c_opt, ll_opt, diagnostics = GaussianDriftModel.optimize_params(returns, vol)
        
        if not diagnostics.get('optimization_successful', True):
            return GaussianPITResult(
                symbol=symbol,
                model_type='gaussian',
                pit_pvalue=0.0,
                ks_statistic=1.0,
                histogram_mad=1.0,
                calibration_grade='F',
                log10_q=float('nan'),
                c=float('nan'),
                phi=0.0,
                nu=0.0,
                alpha=0.0,
                gamma=0.0,
                log_likelihood=float('nan'),
                bic=float('inf'),
                hyvarinen=float(hyv),
                crps=float(crps),
                n_obs=n_obs,
                fit_success=False,
                error=diagnostics.get('error', 'Optimization failed')
            )
        
        # Run filter - returns (mu_filt, P_filt, ll)
        mu_filt, P_filt, ll = GaussianDriftModel.filter(returns, vol, q_opt, c_opt)
        
        # Compute PIT values (no AR(1) drift, so μ_pred = μ_{t-1})
        pit_values = []
        for t in range(len(returns)):
            if t == 0:
                mu_pred = 0.0
                P_pred = 1e-4 + q_opt
            else:
                mu_pred = mu_filt[t-1]  # Random walk (φ=1)
                P_pred = P_filt[t-1] + q_opt
            
            R = c_opt * (vol[t] ** 2)
            S = P_pred + R
            
            innovation = returns[t] - mu_pred
            z = innovation / np.sqrt(max(S, 1e-12))
            
            pit_t = norm.cdf(z)
            pit_values.append(pit_t)
        
        pit_values = np.clip(pit_values, 0.001, 0.999)
        
        # KS test
        ks_result = kstest(pit_values, 'uniform')
        ks_stat = float(ks_result.statistic)
        ks_pvalue = float(ks_result.pvalue)
        
        # Histogram MAD
        hist, _ = np.histogram(pit_values, bins=10, range=(0, 1))
        hist_freq = hist / len(pit_values)
        hist_mad = float(np.mean(np.abs(hist_freq - 0.1)))
        
        # Grade
        if hist_mad < 0.02:
            grade = "A"
        elif hist_mad < 0.05:
            grade = "B"
        elif hist_mad < 0.10:
            grade = "C"
        else:
            grade = "F"
        
        # BIC: 2 parameters (q, c)
        bic = compute_bic(ll, 2, n_obs)
        
        # Compute CRPS and Hyvarinen scores
        forecast_std = np.sqrt(c_opt) * vol
        crps = compute_crps_gaussian(returns, mu_filt, forecast_std)
        hyv = compute_hyvarinen_gaussian(returns, mu_filt, forecast_std)
        
        return GaussianPITResult(
            symbol=symbol,
            model_type='gaussian',
            pit_pvalue=ks_pvalue,
            ks_statistic=ks_stat,
            histogram_mad=hist_mad,
            calibration_grade=grade,
            log10_q=float(np.log10(q_opt)) if q_opt > 0 else float('-inf'),
            c=float(c_opt),
            phi=1.0,  # Random walk
            nu=0.0,
            alpha=0.0,
            gamma=0.0,
            log_likelihood=float(ll),
            bic=float(bic),
            hyvarinen=float(hyv),
            crps=float(crps),
            n_obs=n_obs,
            fit_success=True,
        )
    
    except Exception as e:
        return GaussianPITResult(
            symbol=symbol,
            model_type='gaussian',
            pit_pvalue=0.0,
            ks_statistic=1.0,
            histogram_mad=1.0,
            calibration_grade='F',
            log10_q=float('nan'),
            c=float('nan'),
            phi=1.0,
            nu=0.0,
            alpha=0.0,
            gamma=0.0,
            log_likelihood=float('nan'),
            bic=float('inf'),
                hyvarinen=float(hyv),
                crps=float(crps),
            n_obs=n_obs,
            fit_success=False,
            error=str(e)
        )


def print_result_row(result: GaussianPITResult):
    """Print a single result row with all statistics."""
    pit_ok = "OK" if not result.pit_failed else "X"
    mad_ok = "OK" if not result.mad_failed else "X"
    
    model_short = {
        'gaussian': 'Gauss',
        'phi_gaussian': 'φ-Gauss',
        'phi_gaussian_momentum': 'φ-G+Mom',
    }.get(result.model_type, result.model_type[:8])
    
    # Format ν (degrees of freedom) - show '-' for Gaussian models
    nu_str = f"{result.nu:>3.0f}" if result.nu > 0 else "  -"
    
    # Format α (asymmetry) and γ (persistence/VoV) - show '-' for Gaussian
    alpha_str = f"{result.alpha:+.3f}" if abs(result.alpha) > 1e-10 else "   -  "
    gamma_str = f"{result.gamma:.2f}" if result.gamma > 0 else " -  "
    
    # Expanded format matching Student-t output
    print(f"  {result.symbol:<10} log₁₀(q)={result.log10_q:+.2f}  c={result.c:.3f}  ν={nu_str}  "
          f"φ={result.phi:+.2f}  α={alpha_str}  γ={gamma_str}  |  "
          f"BIC={result.bic:>10.1f}  Hyv={result.hyvarinen:>+10.1f}  CRPS={result.crps:.4f}  "
          f"PIT_p={result.pit_pvalue:.4f} {pit_ok}  MAD={result.histogram_mad:.4f} {mad_ok}  Grd={result.calibration_grade}")


def run_full_test(assets: List[str], include_pure_gaussian: bool = True, include_phi_gaussian: bool = True, include_momentum: bool = True):
    """Run full PIT calibration test for Gaussian models."""
    
    print("=" * 80)
    print("GAUSSIAN PIT CALIBRATION TEST")
    print("=" * 80)
    print()
    
    all_results: List[GaussianPITResult] = []
    
    for i, symbol in enumerate(assets, 1):
        print(f"[{i}/{len(assets)}] Processing {symbol}...")
        
        returns, vol = fetch_asset_data(symbol)
        if returns is None or vol is None:
            print(f"  WARNING: No data available for {symbol}, skipping\n")
            continue
        
        print(f"  Data: {len(returns)} observations")
        
        # φ-Gaussian (AR(1) drift)
        if include_phi_gaussian:
            result_phi_g = run_phi_gaussian_pit(symbol, returns, vol, with_momentum=False)
            print_result_row(result_phi_g)
            all_results.append(result_phi_g)
        
        # φ-Gaussian+Momentum (AR(1) drift with momentum augmentation)
        if include_momentum:
            result_phi_g_mom = run_phi_gaussian_pit(symbol, returns, vol, with_momentum=True)
            print_result_row(result_phi_g_mom)
            all_results.append(result_phi_g_mom)
        
        # Pure Gaussian (random walk)
        if include_pure_gaussian:
            result_g = run_gaussian_pit(symbol, returns, vol)
            print_result_row(result_g)
            all_results.append(result_g)
        
        print()
    
    # Summary
    print("=" * 80)
    print("GAUSSIAN PIT SUMMARY")
    print("=" * 80)
    
    # Group by model type
    model_types = []
    if include_phi_gaussian:
        model_types.append('phi_gaussian')
    if include_momentum:
        model_types.append('phi_gaussian_momentum')
    if include_pure_gaussian:
        model_types.append('gaussian')
    
    for model_type in model_types:
        model_results = [r for r in all_results if r.model_type == model_type]
        if not model_results:
            continue
        
        pit_failures = sum(1 for r in model_results if r.pit_failed)
        mad_failures = sum(1 for r in model_results if r.mad_failed)
        total = len(model_results)
        
        model_name = {
            'gaussian': 'Gaussian (random walk)',
            'phi_gaussian': 'φ-Gaussian (AR(1))',
            'phi_gaussian_momentum': 'φ-Gaussian+Momentum',
        }.get(model_type, model_type)
        
        print(f"\n{model_name}:")
        print(f"  Total assets:      {total}")
        print(f"  PIT failures:      {pit_failures} / {total} (p < 0.05)")
        print(f"  MAD failures:      {mad_failures} / {total} (MAD > 0.05)")
        print(f"  PIT failure rate:  {pit_failures / total * 100:.1f}%")
    
    # Detailed table
    print("\n" + "-" * 120)
    print(f"{'Symbol':<10} {'Model':<10} {'log₁₀(q)':>10} {'c':>6} {'φ':>6} {'BIC':>12} {'PIT_p':>8} {'MAD':>6} {'Grd':<4} {'Status':<6}")
    print("-" * 120)
    
    # Sort by PIT p-value
    for result in sorted(all_results, key=lambda x: x.pit_pvalue):
        status = "PASS" if not result.pit_failed else "FAIL"
        model_short = {
            'gaussian': 'Gauss',
            'phi_gaussian': 'φ-Gauss',
            'phi_gaussian_momentum': 'φ-G+Mom',
        }.get(result.model_type, result.model_type[:8])
        
        print(f"{result.symbol:<10} {model_short:<10} {result.log10_q:>10.2f} {result.c:>6.3f} {result.phi:>+6.2f} "
              f"{result.bic:>12.1f} {result.pit_pvalue:>8.4f} {result.histogram_mad:>6.4f}   {result.calibration_grade:<4} {status:<6}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Gaussian PIT Calibration Test')
    parser.add_argument('--quick', action='store_true', help='Quick test with 5 assets')
    parser.add_argument('--full', action='store_true', help='Full test with 22 assets')
    parser.add_argument('--no-gaussian', action='store_true', help='Skip pure Gaussian')
    parser.add_argument('--no-phi-gaussian', action='store_true', help='Skip φ-Gaussian')
    parser.add_argument('--no-momentum', action='store_true', help='Skip momentum models')
    args = parser.parse_args()
    
    if args.quick:
        assets = QUICK_TEST_ASSETS
    else:
        assets = TEST_ASSETS
    
    run_full_test(
        assets, 
        include_pure_gaussian=not args.no_gaussian,
        include_phi_gaussian=not args.no_phi_gaussian,
        include_momentum=not args.no_momentum,
    )


if __name__ == '__main__':
    main()
