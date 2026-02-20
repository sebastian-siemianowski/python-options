#!/usr/bin/env python3
"""Test Unified Student-T PIT Calibration Failures."""

import sys
import os
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

print('Loading test module...')

import numpy as np
import json
from typing import Optional, Tuple, List
from dataclasses import dataclass, asdict

FAILING_ASSETS = [
    'CDNS', 'CRM', 'ADBE', 'ADI', 'ISRG', 'SAIC', 'FDX', 'BKNG',
    'LLY', 'AIG', 'DE', 'HII', 'HEI', 'MET', 'JCI', 'GOOGL',
    'TMO', 'GOOG', 'UNH', 'USB', 'BAH', 'LDOS',
]

QUICK_TEST_ASSETS = ['GOOGL', 'CDNS', 'CRM', 'ISRG', 'LLY']

# All failing assets from make tune output
ALL_FAILING_ASSETS = [
    'CDNS', 'CRM', 'ADBE', 'ADI', 'ISRG', 'SAIC', 'FDX', 'BKNG',
    'LLY', 'AIG', 'DE', 'HII', 'HEI', 'MET', 'JCI', 'GOOGL',
    'TMO', 'GOOG', 'UNH', 'USB', 'BAH', 'LDOS',
]

PIT_PVALUE_THRESHOLD = 0.05
MAD_THRESHOLD = 0.05

# Storage for results to enable comparison after fixes
RESULTS_FILE = 'test_unified_pit_baseline.json'


@dataclass
class PITTestResult:
    symbol: str
    pit_pvalue: float
    ks_statistic: float
    histogram_mad: float
    calibration_grade: str
    log10_q: float
    c: float
    phi: float
    nu: float
    variance_inflation: float
    gamma_vov: float = 0.0
    alpha_asym: float = 0.0
    ms_sensitivity: float = 2.0
    bic: float = float('nan')
    hyvarinen: float = float('nan')
    crps: float = float('nan')
    log_likelihood: float = float('nan')
    n_obs: int = 0
    fit_success: bool = False
    error: Optional[str] = None
    # ELITE additions (February 2026)
    berkowitz_pvalue: float = float('nan')
    berkowitz_lr: float = float('nan')
    pit_autocorr_lag1: float = float('nan')
    ljung_box_pvalue: float = float('nan')
    has_dynamic_misspec: bool = False
    
    @property
    def pit_failed(self) -> bool:
        return self.pit_pvalue < PIT_PVALUE_THRESHOLD
    
    @property
    def mad_failed(self) -> bool:
        return self.histogram_mad > MAD_THRESHOLD


def fetch_asset_data(symbol, start_date='2015-01-01'):
    try:
        from tuning.tune import _download_prices, compute_hybrid_volatility_har
        df = _download_prices(symbol, start_date, None)
        if df is None or df.empty:
            return None
        cols = {c.lower(): c for c in df.columns}
        if 'close' not in cols:
            return None
        px = df[cols['close']]
        if len(px) < 100:
            return None
        log_ret = np.log(px / px.shift(1)).dropna()
        returns = log_ret.values
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
            return None
        return returns, vol
    except Exception as e:
        print(f'Warning: Data fetch error for {symbol}: {e}')
        return None


def fit_unified_model_and_compute_pit(symbol, returns, vol, nu_base=8.0):
    try:
        from models.phi_student_t import PhiStudentTDriftModel
        from tuning.diagnostics import compute_hyvarinen_score_student_t, compute_crps_student_t_inline
        from calibration.model_selection import compute_bic
        
        config, diagnostics = PhiStudentTDriftModel.optimize_params_unified(
            returns, vol, nu_base=nu_base, train_frac=0.7, asset_symbol=symbol
        )
        if not diagnostics.get('success', False):
            return PITTestResult(
                symbol=symbol, pit_pvalue=0.0, ks_statistic=1.0,
                histogram_mad=1.0, calibration_grade='F',
                log10_q=float('nan'), c=float('nan'), phi=float('nan'),
                nu=nu_base, variance_inflation=1.0, gamma_vov=0.0, alpha_asym=0.0,
                ms_sensitivity=2.0, bic=float('nan'), hyvarinen=float('nan'),
                crps=float('nan'), log_likelihood=float('nan'), n_obs=len(returns),
                fit_success=False,
                error=diagnostics.get('error', 'Optimization failed')
            )
        mu_filt, P_filt, mu_pred, S_pred, ll = PhiStudentTDriftModel.filter_phi_unified(
            returns, vol, config
        )
        ks_stat, pit_pvalue, pit_metrics = PhiStudentTDriftModel.pit_ks_unified(
            returns, mu_pred, S_pred, config
        )
        
        # Compute BIC: -2*LL + k*ln(n)
        # Parameters: q, c, phi, alpha_asym, gamma_vov, ms_sensitivity = 6 core params
        n_params = 6
        n_obs = len(returns)
        bic = compute_bic(ll, n_params, n_obs)
        
        # Compute Hyvarinen score for Student-t
        # Need scale (sigma), not variance. Convert S_pred to scale.
        nu = config.nu_base
        if nu > 2:
            sigma_pred = np.sqrt(S_pred * (nu - 2) / nu)
        else:
            sigma_pred = np.sqrt(S_pred)
        sigma_pred = np.maximum(sigma_pred, 1e-10)
        
        try:
            hyvarinen = compute_hyvarinen_score_student_t(returns, mu_pred, sigma_pred, nu)
        except Exception:
            hyvarinen = float('nan')
        
        # Compute CRPS for Student-t
        try:
            crps = compute_crps_student_t_inline(returns, mu_pred, sigma_pred, nu)
        except Exception:
            crps = float('nan')
        
        return PITTestResult(
            symbol=symbol,
            pit_pvalue=float(pit_pvalue),
            ks_statistic=float(ks_stat),
            histogram_mad=float(pit_metrics.get('histogram_mad', 1.0)),
            calibration_grade=str(pit_metrics.get('calibration_grade', 'F')),
            log10_q=float(np.log10(config.q)) if config.q > 0 else float('-inf'),
            c=float(config.c),
            phi=float(config.phi),
            nu=float(config.nu_base),
            variance_inflation=float(getattr(config, 'variance_inflation', 1.0)),
            gamma_vov=float(getattr(config, 'gamma_vov', 0.0)),
            alpha_asym=float(getattr(config, 'alpha_asym', 0.0)),
            ms_sensitivity=float(getattr(config, 'ms_sensitivity', 2.0)),
            bic=float(bic),
            hyvarinen=float(hyvarinen),
            crps=float(crps),
            log_likelihood=float(ll),
            n_obs=n_obs,
            fit_success=True,
            # ELITE additions
            berkowitz_pvalue=float(pit_metrics.get('berkowitz_pvalue', float('nan'))) if pit_metrics.get('berkowitz_pvalue') is not None else float('nan'),
            berkowitz_lr=float(pit_metrics.get('berkowitz_lr', float('nan'))) if pit_metrics.get('berkowitz_lr') is not None else float('nan'),
            pit_autocorr_lag1=float(pit_metrics.get('pit_autocorr_lag1', float('nan'))) if pit_metrics.get('pit_autocorr_lag1') is not None else float('nan'),
            ljung_box_pvalue=float(pit_metrics.get('ljung_box_pvalue', float('nan'))) if pit_metrics.get('ljung_box_pvalue') is not None else float('nan'),
            has_dynamic_misspec=bool(pit_metrics.get('has_dynamic_misspec', False)),
        )
    except Exception as e:
        return PITTestResult(
            symbol=symbol, pit_pvalue=0.0, ks_statistic=1.0,
            histogram_mad=1.0, calibration_grade='F',
            log10_q=float('nan'), c=float('nan'), phi=float('nan'),
            nu=nu_base, variance_inflation=1.0, gamma_vov=0.0, alpha_asym=0.0,
            ms_sensitivity=2.0, bic=float('nan'), hyvarinen=float('nan'),
            crps=float('nan'), log_likelihood=float('nan'), n_obs=0,
            fit_success=False,
            error=str(e)
        )


def fit_unified_model_adaptive_nu(symbol, returns, vol):
    """
    ELITE IMPROVEMENT: Adaptive nu selection.
    
    Try multiple nu values and select the one that achieves best KS p-value.
    This accounts for the fact that different assets have different tail behavior.
    
    Mathematical basis: The optimal nu minimizes the KS distance between
    the empirical PIT distribution and U[0,1].
    """
    # Nu grid - covers light tails (nu=20) to heavy tails (nu=4)
    NU_GRID = [4, 6, 8, 10, 12, 20]
    
    best_result = None
    best_pvalue = -1.0
    
    for nu in NU_GRID:
        result = fit_unified_model_and_compute_pit(symbol, returns, vol, nu_base=float(nu))
        if result.fit_success and result.pit_pvalue > best_pvalue:
            best_pvalue = result.pit_pvalue
            best_result = result
    
    if best_result is None:
        # Fallback to nu=8 if all failed
        return fit_unified_model_and_compute_pit(symbol, returns, vol, nu_base=8.0)
    
    return best_result


def print_test_result(result):
    status = 'X' if result.pit_failed else 'OK'
    mad_status = 'X' if result.mad_failed else 'OK'
    # Berkowitz status: p > 0.05 means well-calibrated
    berk_p = result.berkowitz_pvalue if np.isfinite(result.berkowitz_pvalue) else 0.0
    berk_status = 'OK' if berk_p >= 0.05 else 'X'
    if result.fit_success:
        # Line 1: Core parameters
        print(f'  {result.symbol:12s}  log₁₀(q)={result.log10_q:+.2f}  c={result.c:.3f}  '
              f'ν={result.nu:.0f}  φ={result.phi:+.2f}  α={result.alpha_asym:+.3f}  '
              f'γ={result.gamma_vov:.2f}')
        # Line 2: Calibration metrics (BIC, Hyvarinen, CRPS)
        print(f'               β={result.variance_inflation:.3f}  '
              f'BIC={result.bic:+.1f}  Hyv={result.hyvarinen:+.4f}  CRPS={result.crps:.4f}')
        # Line 3: PIT diagnostics (KS, Berkowitz, MAD)
        acf1_str = f'{result.pit_autocorr_lag1:+.3f}' if np.isfinite(result.pit_autocorr_lag1) else 'N/A'
        print(f'               PIT: KS_p={result.pit_pvalue:.4f} {status}  '
              f'Berk_p={berk_p:.4f} {berk_status}  '
              f'MAD={result.histogram_mad:.4f} {mad_status}  '
              f'ACF₁={acf1_str}  '
              f'Grade={result.calibration_grade}')
    else:
        print(f'  {result.symbol:12s}  FIT FAILED: {result.error}')


def test_unified_pit_failures_exist():
    print('TEST: Verifying PIT calibration failures exist for unified Student-t model')
    results = []
    n_pit_failures = 0
    n_mad_failures = 0
    for symbol in QUICK_TEST_ASSETS:
        print(f'Processing {symbol}...')
        data = fetch_asset_data(symbol)
        if data is None:
            print(f'Warning: No data available for {symbol}, skipping')
            continue
        returns, vol = data
        print(f'Data: {len(returns)} observations')
        result = fit_unified_model_and_compute_pit(symbol, returns, vol, nu_base=8.0)
        results.append(result)
        print_test_result(result)
        if result.pit_failed:
            n_pit_failures += 1
        if result.mad_failed:
            n_mad_failures += 1
    print('SUMMARY')
    print(f'Assets tested:     {len(results)}')
    print(f'PIT failures:      {n_pit_failures} / {len(results)} (p < 0.05)')
    print(f'MAD failures:      {n_mad_failures} / {len(results)} (MAD > 0.05)')
    if len(results) > 0:
        pit_failure_rate = n_pit_failures / len(results)
        print(f'PIT failure rate:  {pit_failure_rate:.1%}')
        assert pit_failure_rate >= 0.5, (
            f'Expected at least 50% PIT failure rate, got {pit_failure_rate:.1%}. '
            f'Either the data changed or the calibration was improved.'
        )
        print('OK: PIT failures confirmed - test documents current behavior')


def test_adaptive_nu_improvement():
    """
    ELITE TEST: Test adaptive nu selection improvement.
    
    This test verifies that selecting optimal nu per asset improves PIT calibration.
    """
    print('=' * 80)
    print('ELITE TEST: Adaptive Nu Selection')
    print('=' * 80)
    print('Comparing fixed nu=8 vs optimal nu per asset')
    print('')
    
    results_fixed = []
    results_adaptive = []
    
    for symbol in ALL_FAILING_ASSETS:
        print(f'Processing {symbol}...')
        data = fetch_asset_data(symbol)
        if data is None:
            print(f'  Skipping {symbol} - no data')
            continue
        
        returns, vol = data
        
        # Fixed nu=8
        result_fixed = fit_unified_model_and_compute_pit(symbol, returns, vol, nu_base=8.0)
        results_fixed.append(result_fixed)
        
        # Adaptive nu
        result_adaptive = fit_unified_model_adaptive_nu(symbol, returns, vol)
        results_adaptive.append(result_adaptive)
        
        # Compare
        fixed_p = result_fixed.pit_pvalue
        adaptive_p = result_adaptive.pit_pvalue
        fixed_pass = 'PASS' if fixed_p >= 0.05 else 'FAIL'
        adaptive_pass = 'PASS' if adaptive_p >= 0.05 else 'FAIL'
        improvement = adaptive_p - fixed_p
        
        print(f'  {symbol:8s} nu=8: p={fixed_p:.4f} ({fixed_pass})  '
              f'nu={result_adaptive.nu:.0f}: p={adaptive_p:.4f} ({adaptive_pass})  '
              f'Δp={improvement:+.4f}')
    
    # Summary
    n_fixed_pass = sum(1 for r in results_fixed if r.pit_pvalue >= 0.05)
    n_adaptive_pass = sum(1 for r in results_adaptive if r.pit_pvalue >= 0.05)
    
    print('')
    print('=' * 80)
    print('SUMMARY: Adaptive Nu Selection')
    print('=' * 80)
    print(f'Assets tested:           {len(results_fixed)}')
    print(f'Fixed nu=8 passes:       {n_fixed_pass} / {len(results_fixed)} ({100*n_fixed_pass/len(results_fixed):.1f}%)')
    print(f'Adaptive nu passes:      {n_adaptive_pass} / {len(results_adaptive)} ({100*n_adaptive_pass/len(results_adaptive):.1f}%)')
    print(f'Improvement:             +{n_adaptive_pass - n_fixed_pass} assets passing')
    
    # Show optimal nu distribution
    nu_counts = {}
    for r in results_adaptive:
        nu = int(r.nu)
        nu_counts[nu] = nu_counts.get(nu, 0) + 1
    print(f'\nOptimal nu distribution:')
    for nu in sorted(nu_counts.keys()):
        print(f'  nu={nu:2d}: {nu_counts[nu]} assets')
    
    return n_adaptive_pass, n_fixed_pass


def test_full_tuning_all_assets():
    """
    Run full tuning for ALL failing assets and save baseline results.
    
    This creates a baseline JSON file that can be compared after fixes.
    """
    print('=' * 80)
    print('FULL TUNING TEST: Running unified model on all failing assets')
    print('=' * 80)
    
    results = []
    n_pit_failures = 0
    n_mad_failures = 0
    
    for i, symbol in enumerate(ALL_FAILING_ASSETS):
        print(f'\n[{i+1}/{len(ALL_FAILING_ASSETS)}] Processing {symbol}...')
        
        data = fetch_asset_data(symbol)
        if data is None:
            print(f'  WARNING: No data available for {symbol}, skipping')
            results.append({
                'symbol': symbol,
                'fit_success': False,
                'error': 'No data available'
            })
            continue
        
        returns, vol = data
        print(f'  Data: {len(returns)} observations')
        
        result = fit_unified_model_and_compute_pit(symbol, returns, vol, nu_base=8.0)
        
        result_dict = {
            'symbol': result.symbol,
            'pit_pvalue': result.pit_pvalue,
            'ks_statistic': result.ks_statistic,
            'histogram_mad': result.histogram_mad,
            'calibration_grade': result.calibration_grade,
            'log10_q': result.log10_q if not np.isnan(result.log10_q) else None,
            'c': result.c if not np.isnan(result.c) else None,
            'phi': result.phi if not np.isnan(result.phi) else None,
            'nu': result.nu,
            'alpha_asym': result.alpha_asym,
            'gamma_vov': result.gamma_vov,
            'variance_inflation': result.variance_inflation,
            'bic': result.bic if not np.isnan(result.bic) else None,
            'hyvarinen': result.hyvarinen if not np.isnan(result.hyvarinen) else None,
            'crps': result.crps if not np.isnan(result.crps) else None,
            'log_likelihood': result.log_likelihood if not np.isnan(result.log_likelihood) else None,
            'n_obs': result.n_obs,
            'fit_success': result.fit_success,
            'error': result.error,
            'pit_failed': result.pit_failed,
            'mad_failed': result.mad_failed,
        }
        results.append(result_dict)
        
        print_test_result(result)
        
        if result.pit_failed:
            n_pit_failures += 1
        if result.mad_failed:
            n_mad_failures += 1
    
    # Save results to JSON
    baseline = {
        'test_date': '2026-02-19',
        'model': 'phi_student_t_unified_nu_8',
        'total_assets': len(ALL_FAILING_ASSETS),
        'assets_tested': len([r for r in results if r.get('fit_success', False)]),
        'pit_failures': n_pit_failures,
        'mad_failures': n_mad_failures,
        'results': results
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    print('\n' + '=' * 80)
    print('FULL TUNING SUMMARY')
    print('=' * 80)
    print(f'Total assets:      {len(ALL_FAILING_ASSETS)}')
    print(f'Assets tested:     {baseline["assets_tested"]}')
    print(f'PIT failures:      {n_pit_failures} / {baseline["assets_tested"]} (p < 0.05)')
    print(f'MAD failures:      {n_mad_failures} / {baseline["assets_tested"]} (MAD > 0.05)')
    
    if baseline['assets_tested'] > 0:
        pit_failure_rate = n_pit_failures / baseline['assets_tested']
        print(f'PIT failure rate:  {pit_failure_rate:.1%}')
    
    print(f'\nBaseline saved to: {RESULTS_FILE}')
    
    # Print results table with ALL important stats
    print('\n' + '-' * 140)
    print(f'{"Symbol":10s} {"log₁₀(q)":>8s} {"c":>5s} {"ν":>3s} {"φ":>5s} {"α":>6s} {"γ":>4s} '
          f'{"BIC":>9s} {"Hyv":>8s} {"CRPS":>7s} {"PIT_p":>7s} {"MAD":>6s} {"Grd":>3s} {"Status":>6s}')
    print('-' * 140)
    
    for r in sorted(results, key=lambda x: x.get('pit_pvalue', 999) if x.get('fit_success') else 999):
        if r.get('fit_success'):
            q = r['log10_q'] if r['log10_q'] else float('nan')
            c = r['c'] if r['c'] else float('nan')
            phi = r['phi'] if r['phi'] else float('nan')
            nu = r.get('nu', 8)
            alpha = r.get('alpha_asym', 0.0)
            gamma = r.get('gamma_vov', 0.0)
            bic = r.get('bic') if r.get('bic') else float('nan')
            hyv = r.get('hyvarinen') if r.get('hyvarinen') else float('nan')
            crps = r.get('crps') if r.get('crps') else float('nan')
            st = 'FAIL' if r['pit_failed'] else 'PASS'
            print(f'{r["symbol"]:10s} {q:+8.2f} {c:5.3f} {nu:3.0f} {phi:+5.2f} {alpha:+6.3f} {gamma:4.2f} '
                  f'{bic:+9.1f} {hyv:+8.4f} {crps:7.4f} {r["pit_pvalue"]:7.4f} {r["histogram_mad"]:6.4f} '
                  f'{r["calibration_grade"]:>3s} {st:>6s}')
        else:
            print(f'{r["symbol"]:10s} {"---":>8s} {"---":>5s} {"---":>3s} {"---":>5s} {"---":>6s} {"---":>4s} '
                  f'{"---":>9s} {"---":>8s} {"---":>7s} {"---":>7s} {"---":>6s} {"---":>3s} {"ERROR":>6s}')
    
    return baseline


def compare_with_baseline():
    """Compare current results with saved baseline to verify improvements."""
    import os
    
    if not os.path.exists(RESULTS_FILE):
        print(f'No baseline file found at {RESULTS_FILE}')
        print('Run test_full_tuning_all_assets() first to create baseline.')
        return None
    
    with open(RESULTS_FILE, 'r') as f:
        baseline = json.load(f)
    
    print('=' * 80)
    print('COMPARING CURRENT RESULTS WITH BASELINE')
    print('=' * 80)
    print(f'Baseline date: {baseline["test_date"]}')
    print(f'Baseline PIT failures: {baseline["pit_failures"]} / {baseline["assets_tested"]}')
    
    # Run current tests
    current_results = []
    current_pit_failures = 0
    
    for r in baseline['results']:
        if not r.get('fit_success'):
            continue
        
        symbol = r['symbol']
        data = fetch_asset_data(symbol)
        if data is None:
            continue
        
        returns, vol = data
        result = fit_unified_model_and_compute_pit(symbol, returns, vol, nu_base=8.0)
        current_results.append(result)
        if result.pit_failed:
            current_pit_failures += 1
    
    print(f'Current PIT failures:  {current_pit_failures} / {len(current_results)}')
    
    if len(current_results) > 0:
        baseline_rate = baseline['pit_failures'] / baseline['assets_tested']
        current_rate = current_pit_failures / len(current_results)
        improvement = baseline_rate - current_rate
        
        print(f'\nBaseline PIT failure rate: {baseline_rate:.1%}')
        print(f'Current PIT failure rate:  {current_rate:.1%}')
        print(f'Improvement:               {improvement:+.1%}')
        
        if improvement > 0.1:
            print('\n✓ SIGNIFICANT IMPROVEMENT DETECTED!')
        elif improvement > 0:
            print('\n~ Minor improvement detected')
        else:
            print('\n✗ No improvement or regression')
    
    return current_results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        test_full_tuning_all_assets()
    elif len(sys.argv) > 1 and sys.argv[1] == '--compare':
        compare_with_baseline()
    elif len(sys.argv) > 1 and sys.argv[1] == '--adaptive':
        test_adaptive_nu_improvement()
    else:
        print('UNIFIED STUDENT-T PIT CALIBRATION FAILURE TESTS')
        print('Usage:')
        print('  python test_unified_pit_failures.py             # Quick test (5 assets)')
        print('  python test_unified_pit_failures.py --full      # Full test (22 assets)')
        print('  python test_unified_pit_failures.py --adaptive  # Test adaptive nu selection')
        print('  python test_unified_pit_failures.py --compare   # Compare with baseline')
        print('')
        test_unified_pit_failures_exist()
        print('ALL TESTS COMPLETED')