#!/usr/bin/env python3
"""
Comprehensive Model Diagnostic Dashboard for Low-PIT Assets.

Displays ALL metrics for ALL models for assets with PIT p < 0.05,
plus Gold (GC=F) and Silver (SI=F) as reference.

Usage:
  make diag
  .venv/bin/python low_pit_diagnostics.py
"""
import sys
import os
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__) or '.', 'src'))
os.environ['TUNING_QUIET'] = '1'
os.environ['OFFLINE_MODE'] = '1'

import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.rule import Rule
    console = Console(force_terminal=True, color_system="truecolor", width=180)
except ImportError:
    print("rich library required: pip install rich")
    sys.exit(1)

# Low-PIT assets (PIT p < 0.05) sorted by severity + Gold & Silver reference
LOW_PIT_ASSETS = [
    # Critical (p < 0.01)
    'BCAL', 'ERMAY', 'BWXT', 'PLNJPY=X', 'JPYPLN=X', 'GPUS',
    # Warning (p < 0.05)
    'VSH', 'ALMU', 'SIF', 'PSIX', 'SI=F', 'AZBA', 'MSTR', 'SGLP',
    'HWM', 'HII', 'MMM', 'ASML', 'SMCI', 'UPS', 'ALB', 'MRK',
    'PEW', 'GDX', 'ASTS', 'SGDJPY=X', 'JPYSGD=X', 'ABTC', 'PYPL',
    'SNT', 'QS', 'ON', 'AIRI', 'BNKK', 'CRS', 'BNZI', 'EXA',
    'ESLT', 'ACN', 'DFSC', 'ASTC', 'KGC', 'FOUR', 'ADBE', 'OPXS',
    'TFC', 'NVTS', 'GRND', 'XLE', '000660.KS',
]

# Reference assets (always included)
REFERENCE_ASSETS = ['GC=F', 'SI=F']

# Combine, dedup while preserving order
ALL_ASSETS = []
for sym in REFERENCE_ASSETS + LOW_PIT_ASSETS:
    if sym not in ALL_ASSETS:
        ALL_ASSETS.append(sym)


def _pc(val, warn=0.05, good=0.10):
    if not np.isfinite(val): return "dim"
    if val >= good: return "bright_green"
    if val >= warn: return "yellow"
    return "indian_red1"


def _cc(val):
    if not np.isfinite(val): return "dim"
    if val < 0.012: return "bright_green"
    if val < 0.02: return "yellow"
    return "indian_red1"


def _mc(val):
    if not np.isfinite(val): return "dim"
    if val < 0.02: return "bright_green"
    if val < 0.05: return "yellow"
    return "indian_red1"


def _wc(val):
    if val > 0.3: return "bold bright_green"
    if val > 0.1: return "bright_green"
    if val > 0.01: return "white"
    return "dim"


def fetch_data(symbol):
    from tuning.tune import _download_prices, compute_hybrid_volatility_har
    df = _download_prices(symbol, '2015-01-01', None)
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    if 'close' not in cols:
        return None
    px = df[cols['close']]
    log_ret = np.log(px / px.shift(1)).dropna()
    returns = log_ret.values
    if all(c in cols for c in ['open', 'high', 'low', 'close']):
        df_a = df.iloc[1:].copy()
        vol, _ = compute_hybrid_volatility_har(
            open_=df_a[cols['open']].values,
            high=df_a[cols['high']].values,
            low=df_a[cols['low']].values,
            close=df_a[cols['close']].values,
            span=21, annualize=False, use_har=True,
        )
    else:
        vol = log_ret.ewm(span=21).std().values
    mn = min(len(returns), len(vol))
    returns, vol = returns[:mn], vol[:mn]
    ok = np.isfinite(returns) & np.isfinite(vol) & (vol > 0)
    returns, vol = returns[ok], vol[ok]
    if len(returns) < 100:
        return None
    return returns, vol


def fit_models(symbol, returns, vol):
    from tuning.tune import fit_all_models_for_regime
    from tuning.diagnostics import compute_regime_aware_model_weights, CRPS_SCORING_ENABLED

    models = fit_all_models_for_regime(
        returns, vol, prior_log_q_mean=-6.0, prior_lambda=1.0, asset=symbol,
    )
    ok = {m for m in models if models[m].get('fit_success', False)}
    bic_v = {m: models[m].get('bic', float('inf')) for m in ok}
    hyv_v = {m: models[m].get('hyvarinen_score', float('-inf')) for m in ok}
    crps_v = {m: models[m]['crps'] for m in ok
              if models[m].get('crps') is not None and np.isfinite(models[m]['crps'])}
    pit_v = {m: models[m]['pit_ks_pvalue'] for m in ok if models[m].get('pit_ks_pvalue') is not None}
    berk_v = {m: models[m]['berkowitz_pvalue'] for m in ok if models[m].get('berkowitz_pvalue') is not None}
    berk_lr_v = {m: models[m]['berkowitz_lr'] for m in ok if models[m].get('berkowitz_lr') is not None}
    pit_count_v = {m: models[m]['pit_count'] for m in ok if models[m].get('pit_count') is not None}
    mad_v = {m: models[m]['histogram_mad'] for m in ok if models[m].get('histogram_mad') is not None}

    if crps_v and CRPS_SCORING_ENABLED:
        weights, meta = compute_regime_aware_model_weights(
            bic_v, hyv_v, crps_v,
            pit_pvalues=pit_v, berk_pvalues=berk_v,
            berkowitz_lr_stats=berk_lr_v, pit_counts=pit_count_v,
            mad_values=mad_v, regime=None,
        )
    else:
        from tuning.diagnostics import compute_bic_model_weights
        weights = compute_bic_model_weights(bic_v)
        meta = {}

    combined = meta.get('combined_scores_standardized', {})
    for m in models:
        models[m]['weight'] = weights.get(m, 0.0)
        cs = combined.get(m)
        models[m]['combined_score'] = float(cs) if cs is not None else float('nan')

    return models, weights, meta


def render_asset(symbol, models, weights, meta, is_reference=False):
    ok_models = sorted(
        [m for m in models if models[m].get('fit_success', False)],
        key=lambda m: -models[m].get('weight', 0.0),
    )
    fail_models = sorted([m for m in models if not models[m].get('fit_success', False)])

    n_obs_str = ""
    for m in ok_models:
        n = models[m].get('n_obs', 0)
        if n:
            n_obs_str = f"  ({n} obs)"
            break

    console.print()
    border_style = "bright_green" if is_reference else "bright_cyan"
    console.print(Rule(style=border_style))
    title = Text()
    if is_reference:
        title.append("  ★ ", style="bright_green")
    else:
        title.append("  ", style="")
    title.append(f"{symbol}", style="bold bright_white")
    if is_reference:
        title.append("  [REFERENCE]", style="bright_green")
    title.append(f"  —  {len(ok_models)} models", style="dim")
    if fail_models:
        title.append(f", {len(fail_models)} failed", style="indian_red1")
    title.append(n_obs_str, style="dim")
    console.print(title)
    console.print(Rule(style=border_style))

    w_used = meta.get('weights_used', {})
    if w_used:
        wt = Text()
        wt.append("  Scoring: ", style="dim")
        for k, v in w_used.items():
            wt.append(f"{k}=", style="dim")
            wt.append(f"{v:.2f}", style="bright_cyan")
            wt.append("  ", style="")
        wt.append(f"({meta.get('scoring_method', '?')})", style="dim")
        berk_method = meta.get('berkowitz_method', 'unknown')
        berk_lambda = meta.get('berkowitz_lambda_cal', 0.0)
        if berk_method == 'lr_normalized':
            wt.append(f"  berk_method=LR/T λ={berk_lambda:.1f}", style="bright_green")
        elif berk_method:
            wt.append(f"  berk_method={berk_method}", style="dim")
        wt.append("  PIT_gate=ON", style="bright_yellow")
        console.print(wt)
        console.print()

    t = Table(
        box=box.SIMPLE_HEAVY, show_header=True, header_style="bold bright_white",
        border_style=border_style, pad_edge=False, padding=(0, 1),
    )

    t.add_column("Model", style="bold", min_width=28, max_width=38)
    t.add_column("BIC", justify="right", min_width=8)
    t.add_column("CRPS", justify="right", min_width=6)
    t.add_column("Hyv", justify="right", min_width=7)
    t.add_column("PIT_p", justify="right", min_width=6)
    t.add_column("Berk", justify="right", min_width=6)
    t.add_column("MAD", justify="right", min_width=6)
    t.add_column("Score", justify="right", min_width=6)
    t.add_column("Wt", justify="right", min_width=6)
    t.add_column("ν", justify="right", min_width=3)
    t.add_column("φ", justify="right", min_width=5)
    t.add_column("P", justify="center", min_width=1)
    t.add_column("C", justify="center", min_width=1)

    winner = ok_models[0] if ok_models else None

    for m in ok_models:
        d = models[m]
        is_w = (m == winner)
        is_u = d.get('unified_model', False)

        bic = d.get('bic', float('nan'))
        crps = d.get('crps', float('nan'))
        hyv = d.get('hyvarinen_score', float('nan'))
        pit_p = d.get('pit_ks_pvalue', float('nan'))
        berk = d.get('berkowitz_pvalue', float('nan'))
        mad = d.get('histogram_mad', float('nan'))
        score = d.get('combined_score', float('nan'))
        w = d.get('weight', 0.0)
        nu = d.get('nu')
        phi = d.get('phi')

        name = m
        if is_u:
            name += " [U]"
        if is_w:
            name += " ★"
        ns = "bold bright_green" if is_w else ("bright_cyan" if is_u else "white")

        def _f(v, fmt):
            return fmt % v if np.isfinite(v) else "—"

        pit_ok = np.isfinite(pit_p) and pit_p >= 0.05
        berk_ok = np.isfinite(berk) and berk >= 0.05
        both = pit_ok and berk_ok
        pi = "✓" if both else ("~" if (pit_ok or berk_ok) else "✗")
        pic = "bright_green" if both else ("yellow" if (pit_ok or berk_ok) else "indian_red1")

        crps_pass = np.isfinite(crps) and crps < 0.012
        crps_warn = np.isfinite(crps) and 0.012 <= crps < 0.02
        ci = "✓" if crps_pass else ("~" if crps_warn else "✗")
        cic = "bright_green" if crps_pass else ("yellow" if crps_warn else "indian_red1")

        t.add_row(
            Text(name, style=ns),
            Text(_f(bic, "%.1f"), style="cyan"),
            Text(_f(crps, "%.4f"), style=_cc(crps)),
            Text(_f(hyv, "%.0f"), style="white"),
            Text(_f(pit_p, "%.4f"), style=_pc(pit_p)),
            Text(_f(berk, "%.4f"), style=_pc(berk)),
            Text(_f(mad, "%.4f"), style=_mc(mad)),
            Text(_f(score, "%.2f"), style="bold bright_yellow" if is_w else "white"),
            Text("%.3f" % w if w > 0 else "—", style=_wc(w)),
            Text("%.0f" % nu if nu is not None else "—", style="bright_magenta"),
            Text("%+.2f" % phi if phi is not None else "—", style="white"),
            Text(pi, style=pic),
            Text(ci, style=cic),
        )

    console.print(t)

    if fail_models:
        console.print()
        console.print("  [bold indian_red1]Failed:[/bold indian_red1]")
        for m in fail_models:
            err = models[m].get('error', 'unknown')
            if len(str(err)) > 90:
                err = str(err)[:87] + "..."
            console.print(f"    [indian_red1]{m}[/indian_red1]: [dim]{err}[/dim]")

    if winner:
        console.print()
        wd = models[winner]
        ws = Text()
        ws.append("  Winner: ", style="bold bright_green")
        ws.append(winner, style="bold bright_white")
        ws.append(f"  Score={wd.get('combined_score', 0):.3f}", style="dim")
        ws.append(f"  Wt={wd.get('weight', 0):.4f}", style="dim")
        ws.append(f"  PIT={wd.get('pit_ks_pvalue', 0):.4f}", style="dim")
        ws.append(f"  CRPS={wd.get('crps', 0):.4f}", style="dim")
        ws.append(f"  Berk={wd.get('berkowitz_pvalue', 0):.4f}", style="dim")
        console.print(ws)
    console.print()


def process_asset(symbol):
    """Worker function for multiprocessing — fetch data + fit all models for one asset."""
    try:
        data = fetch_data(symbol)
        if data is None:
            return symbol, None, None, None, 'No data'
        returns, vol = data
        models, weights, meta = fit_models(symbol, returns, vol)
        return symbol, models, weights, meta, None
    except Exception as e:
        return symbol, None, None, None, str(e)


def render_pit_summary(all_results, assets, console):
    """Render a cross-asset PIT p-value summary table with models as columns."""
    # Collect all model names across all assets
    all_model_names = set()
    for symbol in assets:
        if symbol not in all_results:
            continue
        models = all_results[symbol]
        for m in models:
            if models[m].get('fit_success', False):
                all_model_names.add(m)

    if not all_model_names:
        return

    # Short model name mapping
    def short_name(m):
        m = m.replace('phi_student_t_unified_nu_', 'U-t')
        m = m.replace('phi_student_t_nu_', 't')
        m = m.replace('kalman_phi_gaussian_unified', 'φ-G-U')
        m = m.replace('kalman_gaussian_unified', 'G-U')
        m = m.replace('kalman_phi_gaussian', 'φ-G')
        m = m.replace('kalman_gaussian', 'G')
        return m

    # Sort models: regular first, then unified
    sorted_models = sorted(all_model_names, key=lambda m: (
        0 if 'unified' not in m else 1,
        0 if 'gaussian' in m.lower() else 1,
        m,
    ))

    console.print()
    console.print(Rule(style="bright_cyan"))
    section = Text()
    section.append("  📊  ", style="bold bright_yellow")
    section.append("PIT SUMMARY — ALL MODELS × ALL ASSETS", style="bold bright_white")
    console.print(section)
    console.print(Rule(style="bright_cyan"))
    console.print()

    t = Table(
        box=box.SIMPLE_HEAVY, show_header=True, header_style="bold bright_white",
        border_style="bright_cyan", pad_edge=False, padding=(0, 1),
    )

    t.add_column("Asset", style="bold", min_width=10, max_width=14)
    for m in sorted_models:
        sn = short_name(m)
        t.add_column(sn, justify="right", min_width=5, max_width=7)
    t.add_column("Best", justify="right", min_width=6, style="bold")
    t.add_column("Winner", style="bold bright_green", min_width=8, max_width=14)

    for symbol in assets:
        if symbol not in all_results:
            # Failed asset
            row_vals = [Text(symbol, style="indian_red1")]
            for _ in sorted_models:
                row_vals.append(Text("—", style="dim"))
            row_vals.append(Text("—", style="dim"))
            row_vals.append(Text("—", style="dim"))
            t.add_row(*row_vals)
            continue

        models = all_results[symbol]
        is_ref = symbol in REFERENCE_ASSETS
        sym_style = "bright_green" if is_ref else "bright_white"

        best_pit = -1.0
        best_model = "—"
        row_vals = [Text(symbol, style=sym_style)]

        for m in sorted_models:
            d = models.get(m)
            if d and d.get('fit_success', False):
                pit_p = d.get('pit_ks_pvalue', float('nan'))
                if np.isfinite(pit_p):
                    row_vals.append(Text("%.3f" % pit_p, style=_pc(pit_p)))
                    if pit_p > best_pit:
                        best_pit = pit_p
                        best_model = short_name(m)
                else:
                    row_vals.append(Text("—", style="dim"))
            else:
                row_vals.append(Text("—", style="dim"))

        row_vals.append(Text("%.3f" % best_pit if best_pit >= 0 else "—", style=_pc(best_pit)))
        row_vals.append(Text(best_model, style="bright_green" if best_pit >= 0.05 else "indian_red1"))
        t.add_row(*row_vals)

    console.print(t)
    console.print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Low-PIT Model Diagnostics")
    parser.add_argument('--critical-only', action='store_true',
                        help='Only show critical assets (PIT p < 0.01)')
    parser.add_argument('--assets', type=str, default=None,
                        help='Comma-separated list of specific assets to diagnose')
    parser.add_argument('--no-reference', action='store_true',
                        help='Skip Gold/Silver reference assets')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')
    parser.add_argument('--pit-only', action='store_true',
                        help='Only show PIT summary table (skip per-asset details)')
    args = parser.parse_args()

    # Determine asset list
    if args.assets:
        assets = [a.strip() for a in args.assets.split(',')]
    elif args.critical_only:
        assets = ['BCAL', 'ERMAY', 'BWXT', 'PLNJPY=X', 'JPYPLN=X', 'GPUS']
        if not args.no_reference:
            for ref in REFERENCE_ASSETS:
                if ref not in assets:
                    assets.insert(0, ref)
    else:
        assets = list(ALL_ASSETS)

    if args.no_reference:
        assets = [a for a in assets if a not in REFERENCE_ASSETS]

    n_critical = 6
    n_warning = len(LOW_PIT_ASSETS) - n_critical
    n_total = len(assets)

    console.print()
    if args.pit_only:
        console.print(Panel(
            "[bold bright_white]PIT SUMMARY — LOW PIT ASSETS[/bold bright_white]\n"
            f"[dim]{n_total} assets · Fitting all models for PIT p-values...[/dim]",
            border_style="bright_cyan", padding=(1, 4),
        ))
    else:
        console.print(Panel(
            "[bold bright_white]COMPREHENSIVE MODEL DIAGNOSTICS — LOW PIT ASSETS[/bold bright_white]\n"
            f"[dim]{n_total} assets · {n_critical} critical (p<0.01) · {n_warning} warning (p<0.05) · All models · All metrics[/dim]",
            border_style="bright_cyan", padding=(1, 4),
        ))

    # Determine parallelism
    n_workers = args.workers or max(1, (mp.cpu_count() or 4) - 1)
    use_parallel = not args.no_parallel and n_total > 1

    if use_parallel:
        console.print(f"  [dim]Using {n_workers} parallel workers (multiprocessing)...[/dim]")
        console.print()

    all_results = {}  # symbol -> models dict (for PIT summary)
    failed_assets = []
    completed = 0

    if use_parallel:
        # Parallel execution with ProcessPoolExecutor
        mp_context = mp.get_context('spawn')
        futures_map = {}
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
            for symbol in assets:
                future = executor.submit(process_asset, symbol)
                futures_map[future] = symbol

            for future in as_completed(futures_map):
                symbol = futures_map[future]
                completed += 1
                try:
                    sym, models, weights, meta, error = future.result()
                    is_ref = sym in REFERENCE_ASSETS
                    if error:
                        console.print(f"  [indian_red1][{completed}/{n_total}] {sym}: {error}[/indian_red1]")
                        failed_assets.append(sym)
                    else:
                        console.print(f"  [bright_green][{completed}/{n_total}] {sym} ✓[/bright_green]")
                        all_results[sym] = models
                        if not args.pit_only:
                            render_asset(sym, models, weights, meta, is_reference=is_ref)
                except Exception as e:
                    console.print(f"  [indian_red1][{completed}/{n_total}] {symbol}: {e}[/indian_red1]")
                    failed_assets.append(symbol)
    else:
        # Sequential execution
        for symbol in assets:
            is_ref = symbol in REFERENCE_ASSETS
            completed += 1
            label = f"[{completed}/{n_total}]"
            if is_ref:
                console.print(f"  [bold bright_green]{label} Loading {symbol} (reference)...[/bold bright_green]")
            else:
                console.print(f"  [bold bright_cyan]{label} Loading {symbol}...[/bold bright_cyan]")
            data = fetch_data(symbol)
            if data is None:
                console.print(f"  [indian_red1]No data for {symbol}[/indian_red1]")
                failed_assets.append(symbol)
                continue
            returns, vol = data
            console.print(f"  [dim]{len(returns)} observations[/dim]")
            console.print(f"  [bold bright_cyan]Fitting all models...[/bold bright_cyan]")
            try:
                models, weights, meta = fit_models(symbol, returns, vol)
                all_results[symbol] = models
                if not args.pit_only:
                    render_asset(symbol, models, weights, meta, is_reference=is_ref)
            except Exception as e:
                console.print(f"  [indian_red1]Error fitting {symbol}: {e}[/indian_red1]")
                failed_assets.append(symbol)

    # PIT Summary table across all assets
    if all_results:
        render_pit_summary(all_results, assets, console)

    # Summary
    console.print()
    console.print(Rule(style="dim"))
    summary = Text()
    summary.append("  Summary: ", style="bold bright_white")
    summary.append(f"{completed - len(failed_assets)}/{n_total} completed", style="bright_green")
    if failed_assets:
        summary.append(f", {len(failed_assets)} failed: ", style="indian_red1")
        summary.append(", ".join(failed_assets), style="indian_red1")
    console.print(summary)
    console.print(Rule(style="dim"))
    console.print("  [dim]Done.[/dim]")
    console.print()


if __name__ == '__main__':
    main()
