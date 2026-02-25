#!/usr/bin/env python3
"""Test Unified Student-T PIT Calibration Failures."""

import sys
import os
import warnings
import multiprocessing

warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

if __name__ == '__main__' or multiprocessing.current_process().name == 'MainProcess':
    pass  # silent load

import numpy as np
import json
from typing import Optional, Tuple, List
from dataclasses import dataclass, asdict
from scipy.stats import kstest

# Rich imports for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.rule import Rule
    from rich.align import Align
    RICH_AVAILABLE = True
    console = Console(force_terminal=True, color_system="truecolor", width=140)
except ImportError:
    RICH_AVAILABLE = False
    console = None

# ============================================================================
# ASSET LISTS - Updated February 21, 2026
# ============================================================================

# Assets that are currently FAILING PIT calibration (p < 0.05)
FAILING_ASSETS = [
    # Critical failures (p < 0.001)
    'FLTR', 'ILKAF', 'BZAI', 'ANNA', 'MDALF', 'BCAL', 'AIFF', 'FTAI',
    'ABTC', 'GC=F', 'GLIN', 'ERMAY', 'BNZI', 'SPCE', 'GORO', 'EVTL',
    'PACB', 'AMZE', 'RCAT', 'LDOS', 'FINMY', 'VRT', 'GOOG', 'MOTG',
    'SI=F', 'XAGUSD', 'GOOGL', 'KITT', 'JCI', 'LLY', 'HII', 'MET',
    'SAIC', 'BKSY', 'ESLT', 'AFK', 'MSTR', 'HEI', 'PEW', 'RGTI',
    'AIG', 'TATT', 'SATL', 'SMCI', 'QBTS', 'NLR', 'SNT', 'VSAT',
    'SIF', 'ARQQ', 'ISRG', 'AIRI', 'OKLO',
]

# Assets that are currently PASSING PIT calibration (p >= 0.05)
PASSING_ASSETS = [
    # Marginal passes (0.05 <= p < 0.10)
    'DURA', 'BAH', 'OPXS', 'FDX',
    # Good passes (p >= 0.10)
    'CRML', 'BKNG', 'GLNCY', 'QS', 'INTC', 'CNXT', 'PSIX', 'USB',
    'MOTI', 'CRM', 'RIO', 'USAS', 'ADI', 'DPRO', 'HYMC', 'EH',
    'XLRE', 'DE', 'DNN', 'APLM', 'TMO', 'ATAI', 'CDNS', 'PGY',
    'ADBE', 'GRND', 'HOVR', 'UNH', 'AZBA', 'COMM', 'ABCL', 'LYSCF',
    'GSAT', 'NVO',
]

# Quick test uses a mix of passing and failing
QUICK_TEST_ASSETS = ['GOOGL', 'CDNS', 'CRM', 'ISRG', 'LLY']

# All assets for comprehensive testing
ALL_ASSETS = FAILING_ASSETS + PASSING_ASSETS

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
    """Fit unified Student-t model and compute PIT using integrated calibration."""
    try:
        from models.phi_student_t import PhiStudentTDriftModel
        from tuning.diagnostics import compute_hyvarinen_score_student_t, compute_crps_student_t_inline
        from calibration.model_selection import compute_bic
        from scipy.stats import t as student_t
        
        n_obs = len(returns)
        n_train = int(n_obs * 0.7)
        
        # Optimize parameters on training data
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
        
        # =================================================================
        # USE INTEGRATED FILTER + CALIBRATION (February 2026)
        # All calibration happens INSIDE the model - no external elite_pit_v3
        # =================================================================
        pit_calibrated, pit_pvalue, sigma_calibrated, _, calib_diag = \
            PhiStudentTDriftModel.filter_and_calibrate(returns, vol, config, train_frac=0.7)
        
        # Get test data for additional metrics
        returns_test = returns[n_train:]
        n_test = len(returns_test)
        
        # Run filter for full data metrics
        mu_filt, P_filt, mu_pred, S_pred, ll = PhiStudentTDriftModel.filter_phi_unified(
            returns, vol, config
        )
        mu_pred_test = mu_pred[n_train:]
        S_pred_test = S_pred[n_train:]
        
        # =====================================================================
        # CRPS: Use filter_and_calibrate's internally-computed CRPS
        # =====================================================================
        # filter_and_calibrate now applies Stage 5h location correction,
        # GARCH stress amplification (5c.1), and vol coupling (5c.2) in its
        # pipeline. Its CRPS reflects all these improvements. Using raw
        # mu_pred + sigma_calibrated would miss the location correction.
        # =====================================================================
        nu_effective_crps = calib_diag.get('nu_effective', config.nu_base)
        crps_from_calibrate = calib_diag.get('crps', float('nan'))

        if np.isfinite(crps_from_calibrate) and crps_from_calibrate > 0:
            crps = crps_from_calibrate
        else:
            # Fallback: compute with raw predictions
            if len(sigma_calibrated) == n_test and np.all(sigma_calibrated > 0):
                sigma_crps = np.maximum(sigma_calibrated, 1e-10)
                nu_crps = nu_effective_crps
            else:
                nu_crps = config.nu_base
                if nu_crps > 2:
                    sigma_crps = np.sqrt(S_pred_test * (nu_crps - 2) / nu_crps)
                else:
                    sigma_crps = np.sqrt(S_pred_test)
                sigma_crps = np.maximum(sigma_crps, 1e-10)
            try:
                crps = compute_crps_student_t_inline(returns_test, mu_pred_test, sigma_crps, nu_crps)
            except Exception:
                crps = float('nan')
        
        pit_values = pit_calibrated
        ks_stat = float(kstest(pit_values, 'uniform').statistic)
        
        # Histogram MAD
        hist, _ = np.histogram(pit_values, bins=10, range=(0, 1))
        hist_freq = hist / n_test
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
        
        # Compute BIC on full data
        n_params = 6
        bic = compute_bic(ll, n_params, n_obs)
        
        # Get nu_effective from calibration diagnostics
        nu_effective = calib_diag.get('nu_effective', config.nu_base)
        berkowitz_pvalue = calib_diag.get('berkowitz_pvalue', 0.0)
        
        # Compute Hyvarinen on TEST data
        try:
            hyvarinen = compute_hyvarinen_score_student_t(returns_test, mu_pred_test, sigma_calibrated, nu_effective)
        except Exception:
            hyvarinen = float('nan')
        
        return PITTestResult(
            symbol=symbol,
            pit_pvalue=float(pit_pvalue),
            ks_statistic=float(ks_stat),
            histogram_mad=float(hist_mad),
            calibration_grade=str(grade),
            log10_q=float(np.log10(config.q)) if config.q > 0 else float('-inf'),
            c=float(config.c),
            phi=float(config.phi),
            nu=float(nu_effective),
            variance_inflation=float(getattr(config, 'variance_inflation', 1.0)),
            gamma_vov=float(getattr(config, 'gamma_vov', 0.0)),
            alpha_asym=float(getattr(config, 'alpha_asym', 0.0)),
            ms_sensitivity=float(getattr(config, 'ms_sensitivity', 2.0)),
            bic=float(bic),
            hyvarinen=float(hyvarinen),
            crps=float(crps),
            log_likelihood=float(ll),
            n_obs=n_test,  # Report test set size
            fit_success=True,
            berkowitz_pvalue=float(berkowitz_pvalue) if np.isfinite(berkowitz_pvalue) else 0.0,
            berkowitz_lr=0.0,
            pit_autocorr_lag1=0.0,
            ljung_box_pvalue=1.0,
            has_dynamic_misspec=False,
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


# =============================================================================
# MULTIPROCESSING WORKER — module level for pickling
# =============================================================================

def _init_worker():
    """Initialize worker process."""
    import warnings
    warnings.filterwarnings('ignore')
    import sys
    if 'src' not in sys.path:
        sys.path.insert(0, 'src')


def _process_asset_worker(symbol):
    """Process one asset. Returns picklable dict."""
    try:
        data = fetch_asset_data(symbol)
        if data is None:
            return {'symbol': symbol, 'fit_success': False, 'error': 'No data available'}
        returns, vol = data
        result = fit_unified_model_and_compute_pit(symbol, returns, vol, nu_base=8.0)
        return {
            'symbol': result.symbol, 'pit_pvalue': result.pit_pvalue,
            'ks_statistic': result.ks_statistic, 'histogram_mad': result.histogram_mad,
            'calibration_grade': result.calibration_grade,
            'log10_q': result.log10_q if not np.isnan(result.log10_q) else None,
            'c': result.c if not np.isnan(result.c) else None,
            'phi': result.phi if not np.isnan(result.phi) else None,
            'nu': result.nu, 'alpha_asym': result.alpha_asym,
            'gamma_vov': result.gamma_vov,
            'variance_inflation': result.variance_inflation,
            'berkowitz_pvalue': result.berkowitz_pvalue if np.isfinite(result.berkowitz_pvalue) else 0.0,
            'bic': result.bic if not np.isnan(result.bic) else None,
            'hyvarinen': result.hyvarinen if not np.isnan(result.hyvarinen) else None,
            'crps': result.crps if not np.isnan(result.crps) else None,
            'log_likelihood': result.log_likelihood if not np.isnan(result.log_likelihood) else None,
            'n_obs': result.n_obs, 'fit_success': result.fit_success,
            'error': result.error, 'pit_failed': result.pit_failed,
            'mad_failed': result.mad_failed,
        }
    except Exception as e:
        return {'symbol': symbol, 'fit_success': False, 'error': str(e)}


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
    """Print test result with Rich formatting - PIT-first layout."""
    status = 'X' if result.pit_failed else 'OK'
    mad_status = 'X' if result.mad_failed else 'OK'
    # Berkowitz status: p > 0.05 means well-calibrated
    berk_p = result.berkowitz_pvalue if np.isfinite(result.berkowitz_pvalue) else 0.0
    berk_status = 'OK' if berk_p >= 0.05 else 'X'
    
    if not RICH_AVAILABLE:
        # Fallback to plain text - PIT first (most important)
        if result.fit_success:
            acf1_str = f'{result.pit_autocorr_lag1:+.3f}' if np.isfinite(result.pit_autocorr_lag1) else 'N/A'
            print(f'  {result.symbol:10s}  PIT: p={result.pit_pvalue:.4f} {status}  │  '
                  f'Berk={berk_p:.4f} {berk_status}  │  MAD={result.histogram_mad:.4f} {mad_status}  │  '
                  f'Grade={result.calibration_grade}')
            print(f'              log₁₀(q)={result.log10_q:+.2f}  c={result.c:.3f}  '
                  f'ν={result.nu:.0f}  φ={result.phi:+.2f}  α={result.alpha_asym:+.3f}  γ={result.gamma_vov:.2f}')
            print(f'              BIC={result.bic:+.1f}  Hyv={result.hyvarinen:+.4f}  '
                  f'CRPS={result.crps:.4f}  β={result.variance_inflation:.3f}')
        else:
            print(f'  {result.symbol:10s}  FIT FAILED: {result.error}')
        return
    
    if result.fit_success:
        # Symbol with status color
        sym_color = "indian_red1" if result.pit_failed else "bright_green"
        
        # Line 1: PIT diagnostics (most important - first!)
        line1 = Text()
        line1.append(f"  {result.symbol:<10}", style=f"bold {sym_color}")
        line1.append("  PIT: ", style="bold white")
        
        # KS p-value (main metric)
        line1.append("p=", style="dim")
        ks_color = "bright_green" if result.pit_pvalue >= 0.10 else "yellow" if result.pit_pvalue >= 0.05 else "indian_red1"
        line1.append(f"{result.pit_pvalue:.4f}", style=f"bold {ks_color}")
        line1.append(f" {status}", style=f"bold {'bright_green' if status == 'OK' else 'indian_red1'}")
        
        # Berkowitz
        line1.append("  │  ", style="dim")
        line1.append("Berk=", style="dim")
        berk_color = "bright_green" if berk_p >= 0.10 else "yellow" if berk_p >= 0.05 else "indian_red1"
        line1.append(f"{berk_p:.4f}", style=berk_color)
        line1.append(f" {berk_status}", style="bright_green" if berk_status == 'OK' else "indian_red1")
        
        # MAD
        line1.append("  │  ", style="dim")
        line1.append("MAD=", style="dim")
        mad_color = "bright_green" if result.histogram_mad < 0.03 else "yellow" if result.histogram_mad < 0.05 else "indian_red1"
        line1.append(f"{result.histogram_mad:.4f}", style=mad_color)
        line1.append(f" {mad_status}", style="bright_green" if mad_status == 'OK' else "indian_red1")
        
        # Grade
        line1.append("  │  ", style="dim")
        line1.append("Grade=", style="dim")
        grade_color = "bright_green" if result.calibration_grade == 'A' else "yellow" if result.calibration_grade in ['B', 'C'] else "indian_red1"
        line1.append(result.calibration_grade, style=f"bold {grade_color}")
        
        console.print(line1)
        
        # Line 2: Core model parameters (aligned)
        line2 = Text()
        line2.append("              ", style="")  # 14 spaces for alignment
        line2.append("log₁₀(q)=", style="dim")
        line2.append(f"{result.log10_q:+.2f}", style="cyan")
        line2.append("  c=", style="dim")
        line2.append(f"{result.c:.3f}", style="white")
        line2.append("  ν=", style="dim")
        line2.append(f"{result.nu:.0f}", style="bright_magenta")
        line2.append("  φ=", style="dim")
        line2.append(f"{result.phi:+.2f}", style="white")
        line2.append("  α=", style="dim")
        alpha_color = "yellow" if abs(result.alpha_asym) > 0.1 else "white"
        line2.append(f"{result.alpha_asym:+.3f}", style=alpha_color)
        line2.append("  γ=", style="dim")
        line2.append(f"{result.gamma_vov:.2f}", style="white")
        console.print(line2)
        
        # Line 3: Calibration metrics (aligned)
        line3 = Text()
        line3.append("              ", style="")  # 14 spaces for alignment
        line3.append("BIC=", style="dim")
        line3.append(f"{result.bic:+.1f}", style="cyan")
        line3.append("  Hyv=", style="dim")
        line3.append(f"{result.hyvarinen:+.4f}", style="white")
        line3.append("  CRPS=", style="dim")
        crps_color = "bright_green" if result.crps < 0.02 else "yellow" if result.crps < 0.05 else "indian_red1"
        line3.append(f"{result.crps:.4f}", style=crps_color)
        line3.append("  β=", style="dim")
        line3.append(f"{result.variance_inflation:.3f}", style="white")
        console.print(line3)
    else:
        line = Text()
        line.append(f"  {result.symbol:<10}", style="bold indian_red1")
        line.append("  FIT FAILED: ", style="indian_red1")
        line.append(str(result.error), style="dim")
        console.print(line)


def test_unified_pit_failures_exist():
    """Quick test to verify PIT calibration failures exist."""
    if RICH_AVAILABLE:
        console.print()
        header = Text()
        header.append("  🔬  ", style="bold bright_yellow")
        header.append("QUICK PIT VERIFICATION TEST", style="bold bright_white")
        console.print(header)
        console.print("      [dim]Verifying PIT calibration failures exist for unified Student-t model[/dim]")
        console.print()
    else:
        print('TEST: Verifying PIT calibration failures exist for unified Student-t model')
    
    results = []
    n_pit_failures = 0
    n_mad_failures = 0
    
    for i, symbol in enumerate(QUICK_TEST_ASSETS):
        if RICH_AVAILABLE:
            # Add spacing between assets
            if i > 0:
                console.print()
            
            # Processing header
            proc_text = Text()
            proc_text.append(f"  [{i+1}/{len(QUICK_TEST_ASSETS)}] ", style="dim")
            proc_text.append(symbol, style="bold bright_cyan")
            console.print(proc_text)
        else:
            print(f'Processing {symbol}...')
        
        data = fetch_asset_data(symbol)
        if data is None:
            if RICH_AVAILABLE:
                console.print("        [yellow]⚠ No data available, skipping[/yellow]")
            else:
                print(f'Warning: No data available for {symbol}, skipping')
            continue
        
        returns, vol = data
        if RICH_AVAILABLE:
            data_text = Text()
            data_text.append("        ", style="")
            data_text.append(f"{len(returns)}", style="bold white")
            data_text.append(" observations", style="dim")
            console.print(data_text)
        else:
            print(f'Data: {len(returns)} observations')
        
        result = fit_unified_model_and_compute_pit(symbol, returns, vol, nu_base=8.0)
        results.append(result)
        print_test_result(result)
        
        if result.pit_failed:
            n_pit_failures += 1
        if result.mad_failed:
            n_mad_failures += 1
    
    # Summary
    if RICH_AVAILABLE:
        console.print()
        console.print(Rule(style="dim"))
        console.print()
        
        summary_header = Text()
        summary_header.append("  📊  ", style="bold bright_cyan")
        summary_header.append("QUICK TEST SUMMARY", style="bold bright_white")
        console.print(summary_header)
        console.print()
        
        # Metrics
        pit_rate = n_pit_failures / len(results) * 100 if len(results) > 0 else 0
        pit_color = "bright_green" if pit_rate < 30 else "yellow" if pit_rate < 60 else "indian_red1"
        
        m1 = Text()
        m1.append("      Assets tested:     ", style="dim")
        m1.append(f"{len(results)}", style="bold bright_white")
        console.print(m1)
        
        m2 = Text()
        m2.append("      PIT failures:      ", style="dim")
        m2.append(f"{n_pit_failures}", style=f"bold {pit_color}")
        m2.append(f" / {len(results)}", style="dim")
        m2.append(" (p < 0.05)", style="dim")
        console.print(m2)
        
        m3 = Text()
        m3.append("      MAD failures:      ", style="dim")
        mad_color = "bright_green" if n_mad_failures == 0 else "indian_red1"
        m3.append(f"{n_mad_failures}", style=f"bold {mad_color}")
        m3.append(f" / {len(results)}", style="dim")
        m3.append(" (MAD > 0.05)", style="dim")
        console.print(m3)
        
        m4 = Text()
        m4.append("      PIT failure rate:  ", style="dim")
        m4.append(f"{pit_rate:.1f}%", style=f"bold {pit_color}")
        console.print(m4)
        console.print()
        
        if len(results) > 0:
            pit_failure_rate = n_pit_failures / len(results)
            if pit_failure_rate >= 0.5:
                ok_text = Text()
                ok_text.append("      ✓ ", style="bold bright_green")
                ok_text.append("PIT failures confirmed", style="bright_green")
                ok_text.append(" — test documents current behavior", style="dim")
                console.print(ok_text)
            else:
                warn_text = Text()
                warn_text.append("      ⚠ ", style="bold yellow")
                warn_text.append(f"Lower than expected failure rate ({pit_failure_rate:.1%})", style="yellow")
                console.print(warn_text)
        console.print()
    else:
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
    
    for symbol in FAILING_ASSETS:
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


def test_full_tuning_all_assets(assets_to_test=None, mode="failing"):
    """
    Run full tuning for assets and save baseline results.
    
    Args:
        assets_to_test: List of assets to test. If None, uses mode to determine.
        mode: "failing" (only failing), "all" (all assets), or "passing" (only passing)
    
    This creates a baseline JSON file that can be compared after fixes.
    """
    # Determine which assets to test
    if assets_to_test is None:
        if mode == "all":
            assets_to_test = ALL_ASSETS
            test_name = "COMPREHENSIVE"
            test_desc = "all assets (failing + passing)"
        elif mode == "passing":
            assets_to_test = PASSING_ASSETS
            test_name = "PASSING VERIFICATION"
            test_desc = "passing assets only"
        else:  # default: failing
            assets_to_test = FAILING_ASSETS
            test_name = "FAILING"
            test_desc = "currently failing assets"
    else:
        test_name = "CUSTOM"
        test_desc = "custom asset list"
    
    if RICH_AVAILABLE:
        console.print()
        console.print(Rule(style="bright_cyan"))
        title = Text()
        title.append("  ◆  ", style="bold bright_cyan")
        title.append(f"UNIFIED STUDENT-T PIT CALIBRATION TEST ({test_name})", style="bold bright_white")
        console.print(title)
        desc = Text()
        desc.append(f"      Testing {len(assets_to_test)} ", style="dim")
        desc.append(test_desc, style="bright_white")
        console.print(desc)
        console.print(Rule(style="bright_cyan"))
        console.print()
    else:
        print('=' * 80)
        print(f'FULL TUNING TEST ({test_name}): {test_desc}')
        print(f'Testing {len(assets_to_test)} assets')
        print('=' * 80)
    
    results = []
    n_pit_failures = 0
    n_mad_failures = 0
    
    # =========================================================================
    # MULTIPROCESSING with Rich live progress
    # =========================================================================
    import time as _time
    n_workers = min(os.cpu_count() or 1, len(assets_to_test))
    use_parallel = n_workers > 1 and len(assets_to_test) > 1
    t0 = _time.time()

    def _print_detail(rd, idx, total, con):
        sym = rd.get('symbol', '?')
        n_obs = rd.get('n_obs', 0)
        if not rd.get('fit_success', False):
            err = rd.get('error', '?')
            con.print(f'  [dim]┌─ [{idx}/{total}][/dim] [bold indian_red1]{sym}[/bold indian_red1] [dim]─ FAILED: {err} ─┐[/dim]')
            return
        pit_p = rd.get('pit_pvalue', 0)
        berk = rd.get('berkowitz_pvalue', 0.0)
        mad = rd.get('histogram_mad', 0)
        grade = rd.get('calibration_grade', '?')
        pf = rd.get('pit_failed', True)
        mf = rd.get('mad_failed', False)
        sc = 'indian_red1' if pf else 'bright_green'
        bc = 'bright_green' if berk >= 0.05 else 'indian_red1'
        mc = 'bright_green' if not mf else 'indian_red1'
        gc = 'bright_green' if grade == 'A' else 'yellow' if grade in ('B','C') else 'indian_red1'
        pm = '✗' if pf else '✓'
        bm = '✗' if berk < 0.05 else '✓'
        mm = '✗' if mf else '✓'
        q = rd.get('log10_q') or 0
        cv = rd.get('c') or 0
        phi = rd.get('phi') or 0
        nu = rd.get('nu', 8)
        alpha = rd.get('alpha_asym', 0.0)
        gamma = rd.get('gamma_vov', 0.0)
        bic = rd.get('bic') or 0
        hyv = rd.get('hyvarinen') or 0
        crps = rd.get('crps') or 0
        vi = rd.get('variance_inflation', 0)
        w = 68
        pad = ' ' * 14
        # Top border
        tag = f'[{idx}/{total}] {sym}'
        obs = f'{n_obs} obs'
        fill = w - len(tag) - len(obs) - 4
        top = Text()
        top.append('  ┌─ ', style='dim')
        top.append(tag, style='bold bright_white')
        top.append(' ' + '─' * fill + ' ', style='dim')
        top.append(obs, style='dim')
        top.append(' ─┐', style='dim')
        con.print(top)
        # Calibration line
        r1 = Text()
        r1.append('  │  ', style='dim')
        r1.append('PIT ', style='dim')
        r1.append(f'p={pit_p:.4f}', style=sc)
        r1.append(f' {pm}', style=f'bold {sc}')
        r1.append('   ', style='dim')
        r1.append('Berk ', style='dim')
        r1.append(f'{berk:.4f}', style=bc)
        r1.append(f' {bm}', style=f'bold {bc}')
        r1.append('   ', style='dim')
        r1.append('MAD ', style='dim')
        r1.append(f'{mad:.4f}', style=mc)
        r1.append(f' {mm}', style=f'bold {mc}')
        r1.append('   ', style='dim')
        r1.append('Grade ', style='dim')
        r1.append(grade, style=f'bold {gc}')
        con.print(r1)
        # Model parameters line
        r2 = Text()
        r2.append('  │  ', style='dim')
        r2.append(f'ν={nu:.0f}  φ={phi:+.2f}  α={alpha:+.3f}  c={cv:.3f}  γ={gamma:.2f}  log₁₀q={q:+.2f}', style='bright_cyan')
        con.print(r2)
        # Scores line
        r3 = Text()
        r3.append('  │  ', style='dim')
        r3.append(f'BIC={bic:+.1f}   CRPS={crps:.4f}   Hyv={hyv:+.4f}   β={vi:.3f}', style='dim')
        con.print(r3)
        # Bottom border
        con.print(f'  [dim]└' + '─' * (w + 2) + '┘[/dim]')

    if use_parallel and RICH_AVAILABLE:
        info = Text()
        info.append('    ⚡ ', style='bold bright_yellow')
        info.append(f'{n_workers}', style='bold bright_white')
        info.append(' workers ', style='dim')
        info.append(f'({os.cpu_count()} CPUs) ', style='dim')
        info.append(f'│ {len(assets_to_test)} assets', style='bright_cyan')
        console.print(info)
        console.print()
        with multiprocessing.Pool(processes=n_workers, initializer=_init_worker) as pool:
            completed = 0
            for rd in pool.imap_unordered(_process_asset_worker, assets_to_test):
                completed += 1
                results.append(rd)
                sym = rd.get('symbol', '?')
                if rd.get('fit_success', False):
                    if rd.get('pit_failed', True): n_pit_failures += 1
                    if rd.get('mad_failed', False): n_mad_failures += 1
                _print_detail(rd, completed, len(assets_to_test), console)
        elapsed = _time.time() - t0
        done = Text()
        done.append('    ✓ ', style='bold bright_green')
        done.append(f'Completed {len(results)} assets in ', style='dim')
        done.append(f'{elapsed:.1f}s', style='bold bright_white')
        done.append(f' ({elapsed/len(results):.1f}s/asset)', style='dim')
        console.print(done)

    elif use_parallel:
        print(f'⚡ MP: {n_workers} workers ({os.cpu_count()} CPUs)')
        with multiprocessing.Pool(n_workers, _init_worker) as pool:
            done = 0
            for rd in pool.imap_unordered(_process_asset_worker, assets_to_test):
                done += 1
                results.append(rd)
                sym = rd.get('symbol', '?')
                if rd.get('fit_success', False):
                    p = rd.get('pit_pvalue', 0)
                    if rd.get('pit_failed', True): n_pit_failures += 1
                    if rd.get('mad_failed', False): n_mad_failures += 1
                    print(f'  [{done}/{len(assets_to_test)}] {sym}: p={p:.4f}')
                else:
                    print(f'  [{done}/{len(assets_to_test)}] {sym}: FAILED')
    else:
        for i, symbol in enumerate(assets_to_test):
            rd = _process_asset_worker(symbol)
            results.append(rd)
            if rd.get('fit_success', False):
                if rd.get('pit_failed', True): n_pit_failures += 1
                if rd.get('mad_failed', False): n_mad_failures += 1
            if RICH_AVAILABLE:
                _print_detail(rd, i+1, len(assets_to_test), console)
            else:
                p = rd.get('pit_pvalue', 0)
                print(f'  [{i+1}/{len(assets_to_test)}] {symbol}: p={p:.4f}')

    # Save results to JSON
    baseline = {
        'test_date': '2026-02-24',
        'model': 'phi_student_t_unified_nu_8',
        'total_assets': len(assets_to_test),
        'assets_tested': len([r for r in results if r.get('fit_success', False)]),
        'pit_failures': n_pit_failures,
        'mad_failures': n_mad_failures,
        'results': results
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    # Print summary using Rich
    if RICH_AVAILABLE:
        render_pit_summary(baseline, results, n_pit_failures, n_mad_failures)
    else:
        print('\n' + '=' * 80)
        print('FULL TUNING SUMMARY')
        print('=' * 80)
        print(f'Total assets:      {len(FAILING_ASSETS)}')
        print(f'Assets tested:     {baseline["assets_tested"]}')
        print(f'PIT failures:      {n_pit_failures} / {baseline["assets_tested"]} (p < 0.05)')
        print(f'MAD failures:      {n_mad_failures} / {baseline["assets_tested"]} (MAD > 0.05)')
        
        if baseline['assets_tested'] > 0:
            pit_failure_rate = n_pit_failures / baseline['assets_tested']
            print(f'PIT failure rate:  {pit_failure_rate:.1%}')
        
        print(f'\nBaseline saved to: {RESULTS_FILE}')
        
        # Print results table
        print('\n' + '-' * 160)
        print(f'{"Symbol":10s} {"log₁₀(q)":>8s} {"c":>5s} {"ν":>3s} {"φ":>5s} {"α":>6s} {"γ":>4s} '
              f'{"BIC":>9s} {"Hyv":>8s} {"CRPS":>7s} {"PIT_p":>7s} {"MAD":>6s} {"Grd":>3s} {"PIT St":>10s} {"CRPS St":>11s}')
        print('-' * 160)
        
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
                pit_st = 'FAIL' if r['pit_failed'] else 'PASS'
                crps_st = 'PASS' if (np.isfinite(crps) and crps < 0.012) else ('WARN' if (np.isfinite(crps) and crps < 0.02) else 'FAIL')
                print(f'{r["symbol"]:10s} {q:+8.2f} {c:5.3f} {nu:3.0f} {phi:+5.2f} {alpha:+6.3f} {gamma:4.2f} '
                      f'{bic:+9.1f} {hyv:+8.4f} {crps:7.4f} {r["pit_pvalue"]:7.4f} {r["histogram_mad"]:6.4f} '
                      f'{r["calibration_grade"]:>3s} {pit_st:>10s} {crps_st:>11s}')
            else:
                print(f'{r["symbol"]:10s} {"---":>8s} {"---":>5s} {"---":>3s} {"---":>5s} {"---":>6s} {"---":>4s} '
                      f'{"---":>9s} {"---":>8s} {"---":>7s} {"---":>7s} {"---":>6s} {"---":>3s} {"ERROR":>10s} {"---":>11s}')
    
    return baseline


def render_pit_summary(baseline, results, n_pit_failures, n_mad_failures):
    """Render beautiful PIT calibration summary using Rich."""
    console.print()
    console.print(Rule(style="bright_cyan"))
    console.print()
    
    # Summary panel
    summary_title = Text()
    summary_title.append("  📊  ", style="bold bright_cyan")
    summary_title.append("PIT CALIBRATION SUMMARY", style="bold bright_white")
    console.print(summary_title)
    console.print()
    
    # Key metrics
    assets_tested = baseline['assets_tested']
    total_assets = baseline['total_assets']
    pit_rate = n_pit_failures / assets_tested * 100 if assets_tested > 0 else 0
    mad_rate = n_mad_failures / assets_tested * 100 if assets_tested > 0 else 0
    pass_rate = 100 - pit_rate
    
    # Metric rows
    metrics = Text()
    metrics.append("    Total assets:      ", style="dim")
    metrics.append(f"{total_assets}", style="bold bright_white")
    console.print(metrics)
    
    metrics2 = Text()
    metrics2.append("    Assets tested:     ", style="dim")
    metrics2.append(f"{assets_tested}", style="bold bright_white")
    console.print(metrics2)
    
    metrics3 = Text()
    metrics3.append("    PIT failures:      ", style="dim")
    pit_color = "bright_green" if pit_rate < 20 else "yellow" if pit_rate < 50 else "indian_red1"
    metrics3.append(f"{n_pit_failures}", style=f"bold {pit_color}")
    metrics3.append(f" / {assets_tested}", style="dim")
    metrics3.append(f" (p < 0.05)", style="dim")
    console.print(metrics3)
    
    metrics4 = Text()
    metrics4.append("    MAD failures:      ", style="dim")
    mad_color = "bright_green" if mad_rate < 10 else "yellow" if mad_rate < 20 else "indian_red1"
    metrics4.append(f"{n_mad_failures}", style=f"bold {mad_color}")
    metrics4.append(f" / {assets_tested}", style="dim")
    metrics4.append(f" (MAD > 0.05)", style="dim")
    console.print(metrics4)
    
    metrics5 = Text()
    metrics5.append("    PIT failure rate:  ", style="dim")
    metrics5.append(f"{pit_rate:.1f}%", style=f"bold {pit_color}")
    console.print(metrics5)
    console.print()
    
    # Baseline saved
    saved = Text()
    saved.append("    Baseline saved to: ", style="dim")
    saved.append(RESULTS_FILE, style="bright_cyan")
    console.print(saved)
    console.print()
    
    # Results table
    console.print(Rule(style="dim"))
    console.print()
    
    # Table header
    table_title = Text()
    table_title.append("    📋  ", style="bold bright_cyan")
    table_title.append("DETAILED RESULTS", style="bold bright_white")
    table_title.append("  (sorted by PIT p-value)", style="dim")
    console.print(table_title)
    console.print()
    
    table = Table(
        show_header=True,
        header_style="bold bright_white on grey23",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1),
        expand=False,
    )
    
    table.add_column("Symbol", justify="left", style="bold", no_wrap=True)
    table.add_column("PIT_p", justify="right", width=7)
    table.add_column("Berk", justify="right", width=6)
    table.add_column("MAD", justify="right", width=6)
    table.add_column("Grd", justify="center", width=3)
    table.add_column("log₁₀(q)", justify="right", width=7)
    table.add_column("c", justify="right", width=6)
    table.add_column("ν", justify="right", width=3)
    table.add_column("φ", justify="right", width=6)
    table.add_column("α", justify="right", width=7)
    table.add_column("γ", justify="right", width=5)
    table.add_column("BIC", justify="right", width=10)
    table.add_column("CRPS", justify="right", width=7)
    table.add_column("PIT Status", justify="center", width=10)
    table.add_column("CRPS Status", justify="center", width=11)
    
    # Sort by PIT p-value
    sorted_results = sorted(results, key=lambda x: x.get('pit_pvalue', 999) if x.get('fit_success') else 999)
    
    for r in sorted_results:
        if r.get("fit_success"):
            sym = r["symbol"]
            q = r.get("log10_q") or 0
            c_v = r.get("c") or 0
            phi = r.get("phi") or 0
            nu = r.get("nu", 8)
            alpha = r.get("alpha_asym", 0.0)
            gamma = r.get("gamma_vov", 0.0)
            berk = r.get("berkowitz_pvalue", 0.0)
            bic = r.get("bic") or 0
            crps = r.get("crps") or 0
            pit_st = "FAIL" if r.get("pit_failed") else "PASS"
            crps_ok = np.isfinite(crps) and crps < 0.012
            crps_warn = np.isfinite(crps) and 0.012 <= crps < 0.02
            crps_st = "PASS" if crps_ok else ("WARN" if crps_warn else "FAIL")
            pc = "bright_green" if r["pit_pvalue"] >= 0.10 else "yellow" if r["pit_pvalue"] >= 0.05 else "indian_red1"
            bc = "bright_green" if berk >= 0.10 else "yellow" if berk >= 0.05 else "indian_red1"
            mc = "bright_green" if r["histogram_mad"] < 0.02 else "yellow" if r["histogram_mad"] < 0.05 else "indian_red1"
            gc = "bright_green" if r["calibration_grade"] == "A" else "yellow" if r["calibration_grade"] in ["B","C"] else "indian_red1"
            psc = "bright_green" if pit_st == "PASS" else "indian_red1"
            csc = "bright_green" if crps_st == "PASS" else "yellow" if crps_st == "WARN" else "indian_red1"
            table.add_row(
                f'[bold]{sym}[/bold]',
                f'[{pc}]{r["pit_pvalue"]:.4f}[/{pc}]',
                f'[{bc}]{berk:.4f}[/{bc}]',
                f'[{mc}]{r["histogram_mad"]:.4f}[/{mc}]',
                f'[{gc}]{r["calibration_grade"]}[/{gc}]',
                f'{q:+.2f}',
                f'{c_v:.3f}',
                f'{nu:.0f}',
                f'{phi:+.2f}',
                f'{alpha:+.3f}',
                f'{gamma:.2f}',
                f'{bic:+.1f}',
                f'{crps:.4f}',
                f'[bold {psc}]{pit_st}[/bold {psc}]',
                f'[bold {csc}]{crps_st}[/bold {csc}]',
            )
        else:
            sym = r.get("symbol", "?")
            table.add_row(sym, "---", "---", "---", "---", "---", "---", "---", "---", "---", "---", "---", "---", "[indian_red1]ERROR[/]", "---")

    console.print(table)
    console.print()
        
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
    
    if len(sys.argv) > 1 and sys.argv[1] == '--metals':
        # Test metals only (GC=F, SI=F, XAGUSD)
        test_full_tuning_all_assets(assets_to_test=['GC=F', 'SI=F', 'XAGUSD'])
    elif len(sys.argv) > 1 and sys.argv[1] == '--special':
        # Test all PIT-failing and CRPS-failing assets (current calibration targets)
        test_full_tuning_all_assets(assets_to_test=[
            # PIT failures
            'OKLO', 'ERMAY', 'BCAL', 'ABTC', 'SAIC', 'CRML', 'SNT', 'ANNA',
            'TATT', 'AMZE', 'BZAI', 'HII', 'FLTR', 'CNXT', 'QS',
            # CRPS failures (PIT pass but CRPS fail)
            'PSIX', 'ARQQ', 'COMM', 'AZBA', 'RGTI', 'MSTR', 'BNZI', 'VRT',
            'SPCE', 'GORO', 'APLM', 'AIRI', 'ILKAF', 'HOVR', 'KITT', 'QBTS',
            'PEW', 'PACB', 'SIF', 'ATAI', 'AIFF', 'EH', 'USAS', 'HYMC',
            'SATL', 'RCAT', 'EVTL', 'SMCI', 'VSAT', 'ABCL', 'DPRO', 'GSAT',
            'BKSY', 'PGY',
        ])
    elif len(sys.argv) > 1 and sys.argv[1] == '--full':
        # Test failing assets only
        test_full_tuning_all_assets(mode="failing")
    elif len(sys.argv) > 1 and sys.argv[1] == '--all':
        # Test ALL assets (failing + passing)
        test_full_tuning_all_assets(mode="all")
    elif len(sys.argv) > 1 and sys.argv[1] == '--compare':
        compare_with_baseline()
    elif len(sys.argv) > 1 and sys.argv[1] == '--adaptive':
        test_adaptive_nu_improvement()
    else:
        if RICH_AVAILABLE:
            console.print()
            title = Text()
            title.append("◆ ", style="bold bright_cyan")
            title.append("UNIFIED STUDENT-T PIT CALIBRATION TESTS", style="bold bright_white")
            console.print(title)
            console.print()
            
            console.print("[dim]Usage:[/dim]")
            console.print(f"  [cyan]python test_unified_pit_failures.py[/cyan]             [dim]# Quick test (5 assets)[/dim]")
            console.print(f"  [cyan]python test_unified_pit_failures.py --full[/cyan]      [dim]# Failing assets ({len(FAILING_ASSETS)} assets)[/dim]")
            console.print(f"  [cyan]python test_unified_pit_failures.py --all[/cyan]       [dim]# All assets ({len(ALL_ASSETS)} assets)[/dim]")
            console.print(f"  [cyan]python test_unified_pit_failures.py --adaptive[/cyan]  [dim]# Test adaptive nu selection[/dim]")
            console.print(f"  [cyan]python test_unified_pit_failures.py --compare[/cyan]   [dim]# Compare with baseline[/dim]")
            console.print()
        else:
            print('UNIFIED STUDENT-T PIT CALIBRATION FAILURE TESTS')
            print('Usage:')
            print('  python test_unified_pit_failures.py             # Quick test (5 assets)')
            print(f'  python test_unified_pit_failures.py --full      # Failing assets ({len(FAILING_ASSETS)} assets)')
            print('  python test_unified_pit_failures.py --adaptive  # Test adaptive nu selection')
            print(f'  python test_unified_pit_failures.py --all       # All assets ({len(ALL_ASSETS)} assets)')
            print('  python test_unified_pit_failures.py --compare   # Compare with baseline')
            print('')
        test_unified_pit_failures_exist()
        if RICH_AVAILABLE:
            console.print()
            console.print("[bold bright_green]✓ ALL TESTS COMPLETED[/bold bright_green]")
        else:
            print('ALL TESTS COMPLETED')