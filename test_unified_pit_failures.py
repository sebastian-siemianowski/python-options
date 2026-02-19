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
    fit_success: bool
    error: Optional[str] = None
    
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
        config, diagnostics = PhiStudentTDriftModel.optimize_params_unified(
            returns, vol, nu_base=nu_base, train_frac=0.7, asset_symbol=symbol
        )
        if not diagnostics.get('success', False):
            return PITTestResult(
                symbol=symbol, pit_pvalue=0.0, ks_statistic=1.0,
                histogram_mad=1.0, calibration_grade='F',
                log10_q=float('nan'), c=float('nan'), phi=float('nan'),
                nu=nu_base, variance_inflation=1.0, fit_success=False,
                error=diagnostics.get('error', 'Optimization failed')
            )
        mu_filt, P_filt, mu_pred, S_pred, ll = PhiStudentTDriftModel.filter_phi_unified(
            returns, vol, config
        )
        ks_stat, pit_pvalue, pit_metrics = PhiStudentTDriftModel.pit_ks_unified(
            returns, mu_pred, S_pred, config
        )
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
            fit_success=True,
        )
    except Exception as e:
        return PITTestResult(
            symbol=symbol, pit_pvalue=0.0, ks_statistic=1.0,
            histogram_mad=1.0, calibration_grade='F',
            log10_q=float('nan'), c=float('nan'), phi=float('nan'),
            nu=nu_base, variance_inflation=1.0, fit_success=False,
            error=str(e)
        )


def print_test_result(result):
    status = 'X' if result.pit_failed else 'OK'
    mad_status = 'X' if result.mad_failed else 'OK'
    if result.fit_success:
        print(f'  {result.symbol:12s}  log10(q)={result.log10_q:+.2f}  c={result.c:.3f}  '
              f'phi={result.phi:+.2f}  beta={result.variance_inflation:.2f}  |  '
              f'PIT_p={result.pit_pvalue:.4f} {status}  MAD={result.histogram_mad:.4f} {mad_status}  '
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


if __name__ == '__main__':
    print('UNIFIED STUDENT-T PIT CALIBRATION FAILURE TESTS')
    test_unified_pit_failures_exist()
    print('ALL TESTS COMPLETED')


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
            'variance_inflation': result.variance_inflation,
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
    
    # Print results table
    print('\n' + '-' * 100)
    print(f'{"Symbol":12s}  {"log10(q)":>9s}  {"c":>6s}  {"phi":>6s}  {"beta":>5s}  '
          f'{"PIT_p":>8s}  {"MAD":>7s}  {"Grade":>5s}  {"Status":>6s}')
    print('-' * 100)
    
    for r in sorted(results, key=lambda x: x.get('pit_pvalue', 999) if x.get('fit_success') else 999):
        if r.get('fit_success'):
            q = r['log10_q'] if r['log10_q'] else float('nan')
            c = r['c'] if r['c'] else float('nan')
            phi = r['phi'] if r['phi'] else float('nan')
            st = 'FAIL' if r['pit_failed'] else 'PASS'
            print(f'{r["symbol"]:12s}  {q:+9.2f}  {c:6.3f}  {phi:+6.2f}  {r["variance_inflation"]:5.2f}  '
                  f'{r["pit_pvalue"]:8.4f}  {r["histogram_mad"]:7.4f}  {r["calibration_grade"]:>5s}  {st:>6s}')
        else:
            print(f'{r["symbol"]:12s}  {"---":>9s}  {"---":>6s}  {"---":>6s}  {"---":>5s}  '
                  f'{"---":>8s}  {"---":>7s}  {"---":>5s}  {"ERROR":>6s}')
    
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
    else:
        print('UNIFIED STUDENT-T PIT CALIBRATION FAILURE TESTS')
        print('Usage:')
        print('  python test_unified_pit_failures.py          # Quick test (5 assets)')
        print('  python test_unified_pit_failures.py --full   # Full test (22 assets)')
        print('  python test_unified_pit_failures.py --compare # Compare with baseline')
        print('')
        test_unified_pit_failures_exist()
        print('ALL TESTS COMPLETED')
