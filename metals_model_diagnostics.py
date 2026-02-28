#!/usr/bin/env python3
"""
Comprehensive Model Diagnostic Dashboard for Gold & Silver.

Displays ALL metrics for ALL models:
  PIT, Berk, MAD, BIC, CRPS, Hyvarinen, Score, Weight

Usage:
  make metals-diag
  .venv/bin/python metals_model_diagnostics.py
"""
import sys
import os
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__) or '.', 'src'))
os.environ['TUNING_QUIET'] = '1'
os.environ['OFFLINE_MODE'] = '1'

import numpy as np

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

METALS = ['GC=F', 'SI=F']


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


def render_asset(symbol, models, weights, meta):
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
    console.print(Rule(style="bright_cyan"))
    title = Text()
    title.append(f"  {symbol}", style="bold bright_white")
    title.append(f"  —  {len(ok_models)} models", style="dim")
    if fail_models:
        title.append(f", {len(fail_models)} failed", style="indian_red1")
    title.append(n_obs_str, style="dim")
    console.print(title)
    console.print(Rule(style="bright_cyan"))

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
        box=box.ROUNDED, show_header=True, header_style="bold bright_white",
        border_style="bright_cyan", pad_edge=False, padding=(0, 1),
    )

    t.add_column("Model", style="bold", min_width=40)
    t.add_column("BIC", justify="right", min_width=10)
    t.add_column("CRPS", justify="right", min_width=8)
    t.add_column("Hyv", justify="right", min_width=10)
    t.add_column("PIT_p", justify="right", min_width=8)
    t.add_column("Berk", justify="right", min_width=8)
    t.add_column("MAD", justify="right", min_width=8)
    t.add_column("Score", justify="right", min_width=8)
    t.add_column("Weight", justify="right", min_width=8)
    t.add_column("nu", justify="right", min_width=4)
    t.add_column("phi", justify="right", min_width=6)
    t.add_column("c", justify="right", min_width=6)
    t.add_column("alpha", justify="right", min_width=7)
    t.add_column("PIT", justify="center", min_width=3)
    t.add_column("CRP", justify="center", min_width=4)

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
        c_val = d.get('c', float('nan'))
        alpha = d.get('alpha_asym', 0.0)
        if alpha == 0.0:
            alpha = d.get('diagnostics', {}).get('alpha_asym', 0.0) if isinstance(d.get('diagnostics'), dict) else 0.0

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
            Text(_f(hyv, "%.1f"), style="white"),
            Text(_f(pit_p, "%.4f"), style=_pc(pit_p)),
            Text(_f(berk, "%.4f"), style=_pc(berk)),
            Text(_f(mad, "%.4f"), style=_mc(mad)),
            Text(_f(score, "%.3f"), style="bold bright_yellow" if is_w else "white"),
            Text("%.4f" % w if w > 0 else "—", style=_wc(w)),
            Text("%.0f" % nu if nu is not None else "—", style="bright_magenta"),
            Text("%+.2f" % phi if phi is not None else "—", style="white"),
            Text("%.3f" % c_val if np.isfinite(c_val) else "—", style="white"),
            Text("%+.3f" % alpha if np.isfinite(alpha) else "—", style="yellow" if abs(alpha) > 0.1 else "white"),
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


def main():
    console.print()
    console.print(Panel(
        "[bold bright_white]COMPREHENSIVE MODEL DIAGNOSTICS — GOLD & SILVER[/bold bright_white]\n"
        "[dim]All models · All metrics · Regime-aware scoring[/dim]",
        border_style="bright_cyan", padding=(1, 4),
    ))

    for symbol in METALS:
        console.print()
        console.print(f"  [bold bright_cyan]Loading {symbol}...[/bold bright_cyan]")
        data = fetch_data(symbol)
        if data is None:
            console.print(f"  [indian_red1]No data for {symbol}[/indian_red1]")
            continue
        returns, vol = data
        console.print(f"  [dim]{len(returns)} observations[/dim]")
        console.print(f"  [bold bright_cyan]Fitting all models...[/bold bright_cyan]")
        models, weights, meta = fit_models(symbol, returns, vol)
        render_asset(symbol, models, weights, meta)

    console.print(Rule(style="dim"))
    console.print("  [dim]Done.[/dim]")
    console.print()


if __name__ == '__main__':
    main()
