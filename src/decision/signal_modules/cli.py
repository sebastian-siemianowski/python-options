"""
signal_modules/cli -- CLI argument parsing, orchestration, and rendering.

Extracted from signals.py (Story 8.7).

Functions:
    parse_args()                      -- CLI argument parser
    _process_assets_with_retries()    -- parallel processing with bounded retries
    render_regime_model_summary()     -- regime-model selection table
    main()                            -- entry-point orchestration
"""
from __future__ import annotations

import os
import sys
import json
import re
import argparse
import math

# ---------------------------------------------------------------------------
# Path setup -- identical to other signal_modules files
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Directory containing the parent signals.py (src/decision/)
_DECISION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Optional, Tuple  # noqa: E402
from multiprocessing import Pool, cpu_count  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rich.console import Console  # noqa: E402
from rich.text import Text  # noqa: E402

# ---------------------------------------------------------------------------
# Wildcard imports from signal_modules -- bring in all public symbols
# (feature flags, presentation helpers, constants like NOTIONAL_PLN, etc.)
# ---------------------------------------------------------------------------
from decision.signal_modules.config import *  # noqa: F401,F403,E402
from decision.signal_modules.config import (  # noqa: E402
    _resolve_display_name, _resolve_symbol_candidates,
)

from decision.signal_modules.signal_generation import *  # noqa: F401,F403,E402
from decision.signal_modules.asset_processing import process_single_asset  # noqa: E402

# ---------------------------------------------------------------------------
# CLI-specific constants (originally in signals.py module scope)
# ---------------------------------------------------------------------------
DEFAULT_HORIZONS = [1, 3, 7, 21, 63, 126, 252]
DEFAULT_CACHE_PATH = os.path.join("src", "data", "currencies", "fx_plnjpy.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate signals across multiple horizons for PLN/JPY, Gold (PLN), Silver (PLN), Bitcoin (PLN), and MicroStrategy (PLN).")
    p.add_argument("--start", type=str, default="2005-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--horizons", type=str, default=",".join(map(str, DEFAULT_HORIZONS)))
    p.add_argument("--assets", type=str, default=",".join(DEFAULT_ASSET_UNIVERSE), help="Comma-separated Yahoo symbols or friendly names. Metals, FX and USD/EUR/GBP/JPY/CAD/DKK/KRW assets are converted to PLN.")
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--cache-json", type=str, default=DEFAULT_CACHE_PATH, help="Path to auto-write cache JSON (default src/data/currencies/fx_plnjpy.json)")
    p.add_argument("--from-cache", action="store_true", help="Render tables from cache JSON and skip computation")
    p.add_argument("--simple", action="store_true", help="Print an easy-to-read summary with simple explanations.")
    p.add_argument("--t_map", action="store_true", help="Use Student-t mapping based on realized kurtosis for probabilities (default on).")
    p.add_argument("--no_t_map", dest="t_map", action="store_false", help="Disable Student-t mapping; use Normal CDF.")
    p.add_argument("--ci", type=float, default=0.68, help="Two-sided confidence level for expected move bands (default 0.68 i.e., ~1-sigma).")
    # Caption controls for detailed view
    p.add_argument("--no_caption", action="store_true", help="Suppress the long column explanation caption in detailed tables.")
    p.add_argument("--force_caption", action="store_true", help="Force showing the caption for every detailed table.")
    # Diagnostics controls (Level-7 falsifiability)
    p.add_argument("--diagnostics", action="store_true", help="Enable full diagnostics: log-likelihood monitoring, parameter stability tracking, and out-of-sample tests (expensive).")
    p.add_argument("--diagnostics_lite", action="store_true", help="Enable lightweight diagnostics: log-likelihood monitoring and parameter stability (no OOS tests).")
    p.add_argument("--pit-calibration", action="store_true", help="Enable PIT calibration verification: tests if predicted probabilities match actual outcomes (Level-7 requirement, very expensive).")
    p.add_argument("--model-comparison", action="store_true", help="Enable structural model comparison: GARCH vs EWMA, Student-t vs Gaussian, Kalman vs EWMA using AIC/BIC (Level-7 falsifiability).")
    p.add_argument("--validate-kalman", action="store_true", help="🧪 Run Level-7 Kalman validation science: drift reasonableness, predictive likelihood improvement, PIT calibration, and stress-regime behavior analysis.")
    p.add_argument("--validation-plots", action="store_true", help="Generate diagnostic plots for Kalman validation (requires --validate-kalman).")
    p.add_argument("--failures-json", type=str, default=os.path.join(_DECISION_DIR, "fx_failures.json"), help="Where to write failure log (set to '' to disable)")
    p.set_defaults(t_map=True)
    return p.parse_args()


def _process_assets_with_retries(assets: List[str], args: argparse.Namespace, horizons: List[int], max_retries: int = 3):
    """Run asset processing with bounded retries and collect failures.
    Retries only the assets that failed on prior attempts.
    Uses multiprocessing.Pool for true multi-process parallelism (CPU-bound work).
    """
    from rich.rule import Rule
    from rich.align import Align

    console = Console(force_terminal=True, width=140)
    pending = list(dict.fromkeys(a.strip() for a in assets if a and a.strip()))
    successes: List[Dict] = []
    failures: Dict[str, Dict[str, object]] = {}
    processed_canon = set()

    # ═══════════════════════════════════════════════════════════════════════════════
    # EXTRAORDINARY APPLE-QUALITY PROCESSING UX
    # ═══════════════════════════════════════════════════════════════════════════════

    console.print()
    console.print(Rule(style="dim"))
    console.print()

    # Processing header
    header = Text()
    header.append("▸ ", style="bright_cyan")
    header.append("PROCESSING", style="bold white")
    console.print(header)
    console.print()

    # Stats row
    n_workers = min(cpu_count(), len(pending))
    stats = Text()
    stats.append("    ", style="")
    stats.append(f"{len(pending)}", style="bold bright_cyan")
    stats.append(" assets", style="dim")
    stats.append("  ·  ", style="dim")
    stats.append(f"{n_workers}", style="bold white")
    stats.append(" cores", style="dim")
    stats.append("  ·  ", style="dim")
    stats.append(f"{max_retries}", style="white")
    stats.append(" max retries", style="dim")
    console.print(stats)
    console.print()

    attempt = 1
    while attempt <= max_retries and pending:
        n_workers = min(cpu_count(), len(pending))

        # Pass indicator
        pass_text = Text()
        pass_text.append(f"    Pass {attempt}/{max_retries}", style="dim")
        pass_text.append(f"  ·  {len(pending)} pending", style="dim")
        console.print(pass_text)

        work_items = [(asset, args, horizons) for asset in pending]

        # Prefetch prices in bulk on first attempt to reduce Yahoo rate limits
        if attempt == 1 and pending:
            try:
                # Suppress verbose output and symbol tables - they're shown after validation
                download_prices_bulk(pending, start=args.start, end=args.end, progress=False, show_symbol_tables=False)
            except Exception as e:
                console.print(f"    [yellow]⚠[/yellow] [dim]Bulk prefetch failed, using standard fetch[/dim]")

        # Always use multiprocessing.Pool for true multi-process parallelism
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_single_asset, work_items)

        next_pending: List[str] = []
        pass_successes = 0
        for asset, result in zip(pending, results):
            if not result or result.get("status") != "success":
                err = (result or {}).get("error", "unknown")
                tb = (result or {}).get("traceback")
                if tb:
                    tb_lines = [line.strip() for line in str(tb).splitlines() if line.strip()]
                    loc_lines = [ln for ln in tb_lines if ln.startswith("File ")]
                    loc_line = loc_lines[-1] if loc_lines else None
                    if loc_line:
                        err = f"{err} @ {loc_line}"
                try:
                    disp = _resolve_display_name(asset.strip().upper())
                except Exception:
                    disp = asset
                entry = failures.get(asset, {"attempts": 0, "last_error": None, "display_name": disp, "traceback": None})
                entry["attempts"] = int(entry.get("attempts", 0)) + 1
                entry["last_error"] = err
                if tb:
                    entry["traceback"] = tb
                entry["display_name"] = entry.get("display_name") or disp
                failures[asset] = entry
                next_pending.append(asset)
                continue


            canon = result.get("canon") or asset.strip().upper()
            if canon in processed_canon:
                continue
            processed_canon.add(canon)
            successes.append(result)
            pass_successes += 1
            # drop from pending on success; nothing to add to next_pending
            if asset in failures:
                failures.pop(asset, None)

        pending = list(dict.fromkeys(next_pending))

        # Pass result
        if pass_successes > 0:
            console.print(f"    [bright_green]✓[/bright_green] [dim]{pass_successes} succeeded[/dim]")

        # Early exit: if all assets succeeded, skip remaining passes
        if not pending:
            break

        if pending and attempt < max_retries:
            console.print(f"    [yellow]○[/yellow] [dim]{len(pending)} retrying...[/dim]")

        attempt += 1

    console.print()

    # Final status
    if not pending:
        done = Text()
        done.append("    ", style="")
        done.append("✓", style="bold bright_green")
        done.append(f"  {len(successes)} assets processed", style="white")
        console.print(done)
    else:
        done = Text()
        done.append("    ", style="")
        done.append("!", style="bold yellow")
        done.append(f"  {len(successes)} succeeded, {len(pending)} failed", style="white")
        console.print(done)

    console.print()
    console.print(Rule(style="dim"))
    console.print()

    return successes, failures


def render_regime_model_summary(regime_model_tracker: Dict[str, Dict[str, int]], console: Console = None) -> None:
    """
    Render a summary table showing model usage per regime.
    
    This helps identify which models are selected in each market regime,
    useful for understanding model behavior and identifying unused models.
    
    Args:
        regime_model_tracker: Dict[regime_name, Dict[model_short_name, count]]
        console: Rich Console for output
    """
    if console is None:
        console = Console(force_terminal=True, width=140)
    
    if not regime_model_tracker:
        return
    
    from rich.table import Table
    from rich.text import Text
    from rich.rule import Rule
    
    # Count total assets per regime
    regime_totals = {}
    for regime, models in regime_model_tracker.items():
        regime_totals[regime] = sum(models.values())
    
    total_assets = sum(regime_totals.values())
    if total_assets == 0:
        return
    
    # Collect all unique models across regimes
    all_models = set()
    for models in regime_model_tracker.values():
        all_models.update(models.keys())
    
    # Sort models by total usage (most used first)
    model_usage = {}
    for model in all_models:
        model_usage[model] = sum(regime_model_tracker.get(r, {}).get(model, 0) for r in regime_model_tracker)
    sorted_models = sorted(all_models, key=lambda m: model_usage[m], reverse=True)
    
    # Only show models with at least 1% usage overall
    min_usage = max(1, int(total_assets * 0.01))
    visible_models = [m for m in sorted_models if model_usage[m] >= min_usage]
    
    if not visible_models:
        return
    
    # Define regime display order and labels
    # HMM regime names: calm, trending, crisis (sorted by volatility)
    regime_order = ['calm', 'trending', 'crisis', 'LOW_VOL_TREND', 'HIGH_VOL_TREND', 'LOW_VOL_RANGE', 'HIGH_VOL_RANGE', 'CRISIS_JUMP']
    regime_labels = {
        # HMM regime names (primary)
        'calm': 'Calm',
        'trending': 'Trending',
        'crisis': 'Crisis',
        # Legacy regime names (for backwards compatibility)
        'LOW_VOL_TREND': 'Low-Vol Trend',
        'HIGH_VOL_TREND': 'High-Vol Trend',
        'LOW_VOL_RANGE': 'Low-Vol Range',
        'HIGH_VOL_RANGE': 'High-Vol Range',
        'CRISIS_JUMP': 'Crisis/Jump',
        'Unknown': 'Unknown',
    }
    
    # Build the table
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    
    header = Text()
    header.append("▸ ", style="bright_cyan")
    header.append("MODEL SELECTION BY REGIME", style="bold white")
    console.print(header)
    console.print()
    
    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("Model", style="white", justify="left")
    table.add_column("Total", style="bold white", justify="right")
    
    # Add regime columns
    active_regimes = [r for r in regime_order if r in regime_model_tracker]
    # Add any regimes not in the predefined order
    for r in regime_model_tracker:
        if r not in active_regimes:
            active_regimes.append(r)
    
    for regime in active_regimes:
        label = regime_labels.get(regime, regime)
        table.add_column(label, justify="right")
    
    # Add rows for each model
    for model in visible_models:
        total = model_usage[model]
        total_pct = (total / total_assets * 100) if total_assets > 0 else 0
        
        row = [model, f"{total} ({total_pct:.0f}%)"]
        
        for regime in active_regimes:
            count = regime_model_tracker.get(regime, {}).get(model, 0)
            regime_total = regime_totals.get(regime, 1)
            pct = (count / regime_total * 100) if regime_total > 0 else 0
            if count > 0:
                row.append(f"{count} ({pct:.0f}%)")
            else:
                row.append("[dim]—[/dim]")
        
        table.add_row(*row)
    
    # Add totals row
    total_row = ["[bold]Total Assets[/bold]", f"[bold]{total_assets}[/bold]"]
    for regime in active_regimes:
        count = regime_totals.get(regime, 0)
        pct = (count / total_assets * 100) if total_assets > 0 else 0
        total_row.append(f"[bold]{count}[/bold] ({pct:.0f}%)")
    table.add_row(*total_row)
    
    console.print(table)
    console.print()


def main() -> None:
    args = parse_args()
    horizons = sorted({int(x.strip()) for x in args.horizons.split(",") if x.strip()})

    # Fast path: render from cache only (SUMMARY ONLY - no detailed tables)
    if args.from_cache:
        cache_path = args.cache_json or DEFAULT_CACHE_PATH
        if not os.path.exists(cache_path):
            Console().print(f"[indian_red1]Cache not found:[/indian_red1] {cache_path}")
            return
        with open(cache_path, "r") as f:
            payload = json.load(f)
        horizons_cached = payload.get("horizons") or horizons
        assets_cached = payload.get("assets", [])
        summary_rows_cached = payload.get("summary_rows")

        console = Console()

        # Build summary rows from cache (skip detailed tables for compact display)
        if not summary_rows_cached:
            summary_rows_cached = []
            for asset_data in assets_cached:
                sym = asset_data.get("symbol") or ""
                title = asset_data.get("title") or sym
                asset_label = build_asset_display_label(sym, title)
                # Use get_sector() to properly look up sector from SECTOR_MAP
                sector = asset_data.get("sector") or get_sector(sym) or "Other"
                horizon_signals = {}
                for sig in asset_data.get("signals", []):
                    h = sig.get("horizon_days")
                    if h is None:
                        continue
                    horizon_signals[int(h)] = {
                        "label": sig.get("label", "HOLD"),
                        "profit_pln": float(sig.get("profit_pln", 0.0)),
                        "ue_up": float(sig.get("ue_up", 0.0)),
                        "ue_down": float(sig.get("ue_down", 0.0)),
                        "p_up": float(sig.get("p_up", 0.5)),
                        "exp_ret": float(sig.get("exp_ret", 0.0)),
                    }
                nearest_label = next(iter(horizon_signals.values()), {}).get("label", "HOLD")
                summary_rows_cached.append({"asset_label": asset_label, "horizon_signals": horizon_signals, "nearest_label": nearest_label, "sector": sector})
        else:
            # Ensure existing summary_rows have proper sectors and p_up/exp_ret fields
            # This handles old cache format that may be missing these fields
            for row in summary_rows_cached:
                if not row.get("sector"):
                    # Try to extract symbol from asset_label and look up sector
                    label = row.get("asset_label", "")
                    # Extract symbol from "Name (SYM)" format
                    import re
                    match = re.search(r'\(([A-Z0-9.-]+)\)', label)
                    if match:
                        sym = match.group(1)
                        row["sector"] = get_sector(sym) or "Other"
                    else:
                        row["sector"] = "Other"

                # Ensure crash_risk_score exists (default 0 for old cache format)
                if "crash_risk_score" not in row:
                    row["crash_risk_score"] = 0

                # Ensure horizon_signals have p_up and exp_ret (from assets_cached if needed)
                horizon_signals = row.get("horizon_signals", {})
                for h, sig_data in horizon_signals.items():
                    if "p_up" not in sig_data or "exp_ret" not in sig_data:
                        # Try to find from assets_cached
                        for asset_data in assets_cached:
                            for sig in asset_data.get("signals", []):
                                if sig.get("horizon_days") == h or sig.get("horizon_days") == int(h):
                                    if "p_up" not in sig_data:
                                        sig_data["p_up"] = float(sig.get("p_up", 0.5))
                                    if "exp_ret" not in sig_data:
                                        sig_data["exp_ret"] = float(sig.get("exp_ret", 0.0))
                                    break

        try:
            render_sector_summary_tables(summary_rows_cached, horizons_cached)
            # Add high-conviction signals summary for short-term trading
            render_strong_signals_summary(summary_rows_cached, horizons=[1, 3, 7])
            
            # Save high conviction signals to src/data/high_conviction/ with options data
            if HIGH_CONVICTION_STORAGE_AVAILABLE:
                try:
                    result = save_high_conviction_signals(
                        summary_rows_cached, 
                        horizons=[1, 3, 7],
                        fetch_options=True,
                        fetch_prices=True,
                    )
                    if result.get("buy", 0) > 0 or result.get("sell", 0) > 0:
                        Console().print(
                            f"\n[dim]💾 Saved {result['buy']} buy + {result['sell']} sell signals "
                            f"to src/data/high_conviction/[/dim]"
                        )
                        # Generate candlestick charts for strong signals
                        if SIGNAL_CHARTS_AVAILABLE:
                            try:
                                generate_signal_charts(quiet=False)
                            except Exception as chart_e:
                                if os.getenv("DEBUG"):
                                    Console().print(f"[dim]Signal chart error: {chart_e}[/dim]")
                except Exception as hc_e:
                    if os.getenv("DEBUG"):
                        Console().print(f"[dim]High conviction storage error: {hc_e}[/dim]")
            
            # Below SMA50 analysis — table + charts
            if SIGNAL_CHARTS_AVAILABLE and find_below_sma50_stocks is not None:
                try:
                    below_sma50 = find_below_sma50_stocks(summary_rows_cached)
                    if below_sma50:
                        render_below_sma50_table(below_sma50, quiet=False)
                        generate_sma_charts(below_sma50, quiet=False)
                except Exception as sma_e:
                    if os.getenv("DEBUG"):
                        Console().print(f"[dim]SMA50 analysis error: {sma_e}[/dim]")
            
            # Index fund / ETF charts
            if SIGNAL_CHARTS_AVAILABLE and generate_index_charts is not None:
                try:
                    generate_index_charts(quiet=False)
                except Exception as idx_e:
                    if os.getenv("DEBUG"):
                        Console().print(f"[dim]Index chart error: {idx_e}[/dim]")
            
            # Show unified risk dashboard (replaces fragmented risk temperature display)
            if UNIFIED_RISK_DASHBOARD_AVAILABLE:
                try:
                    compute_and_render_unified_risk(
                        start_date="2020-01-01",
                        suppress_output=True,
                        console=Console(),
                        use_parallel=True,  # Use maximum processors for speed
                    )
                except Exception:
                    # Fallback to legacy
                    if RISK_TEMPERATURE_AVAILABLE:
                        try:
                            risk_temp_result = get_cached_risk_temperature(
                                start_date="2020-01-01",
                                notional=NOTIONAL_PLN,
                                estimated_gap_risk=0.03,
                            )
                            render_risk_temperature_summary(risk_temp_result)
                        except Exception:
                            pass
            elif RISK_TEMPERATURE_AVAILABLE:
                try:
                    risk_temp_result = get_cached_risk_temperature(
                        start_date="2020-01-01",
                        notional=NOTIONAL_PLN,
                        estimated_gap_risk=0.03,
                    )
                    render_risk_temperature_summary(risk_temp_result)
                except Exception:
                    pass  # Silently skip if risk temp fails
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not print summary tables from cache: {e}")
        return

    # Parse assets
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]

    console = Console(force_terminal=True, width=140)

    # ═══════════════════════════════════════════════════════════════════════════════
    # VALIDATION PHASE - Apple-quality UX
    # ═══════════════════════════════════════════════════════════════════════════════
    from rich.rule import Rule

    console.print()
    console.print(Rule(style="dim"))
    console.print()

    validation_header = Text()
    validation_header.append("▸ ", style="bright_cyan")
    validation_header.append("VALIDATION", style="bold white")
    console.print(validation_header)
    console.print()

    validation_stats = Text()
    validation_stats.append("    ", style="")
    validation_stats.append(f"{len(assets)}", style="bold bright_cyan")
    validation_stats.append(" assets requested", style="dim")
    console.print(validation_stats)
    console.print()

    for a in assets:
        try:
            _resolve_symbol_candidates(a)
        except Exception as e:
            console.print(f"    [yellow]⚠[/yellow] [dim]{a}: {e}[/dim]")

    # Print symbol resolution table right after validation (before processing starts)
    print_symbol_tables()

    all_blocks = []  # for JSON export
    csv_rows_simple = []  # for CSV simple export
    csv_rows_detailed = []  # for CSV detailed export
    summary_rows = []  # for summary table across assets
    regime_model_tracker = {}  # Dict[regime_name, Dict[model_short_name, count]] for end-of-run summary

    # =========================================================================
    # RETRYING PARALLEL PROCESSING: Compute features/signals with bounded retries
    # =========================================================================
    success_results, failures = _process_assets_with_retries(assets, args, horizons, max_retries=3)

    # =========================================================================
    # SEQUENTIAL DISPLAY & AGGREGATION: Process results in order with console output
    # =========================================================================
    caption_printed = False
    processed_syms = set()

    for result in success_results:
        # Handle None or error results
        if result is None:
            continue

        if result.get("status") == "error":
            asset = result.get("asset", "unknown")
            error = result.get("error", "unknown")
            console.print(f"[indian_red1]Warning:[/indian_red1] Failed to process {asset}: {error}")
            if os.getenv('DEBUG'):
                traceback_info = result.get("traceback", "")
                if traceback_info:
                    console.print(f"[dim]{traceback_info}[/dim]")
            continue

        if result.get("status") != "success":
            continue

        # Extract computed results from worker
        asset = result["asset"]
        canon = result["canon"]
        title = result["title"]
        px = result["px"]
        feats = result["feats"]
        sigs = result["sigs"]
        thresholds = result["thresholds"]
        diagnostics = result["diagnostics"]
        last_close = result["last_close"]
        enrichment = result.get("enrichment", {})

        # De-duplicate check
        if canon in processed_syms:
            console.print(f"[yellow]Skipping duplicate:[/yellow] {title} (from input '{asset}')")
            continue
        processed_syms.add(canon)

        # Print table for this asset
        if args.simple:
            explanations = render_simplified_signal_table(asset, title, sigs, px, feats)
        else:
            # Determine caption policy for detailed view
            if args.force_caption:
                show_caption = True
            elif args.no_caption:
                show_caption = False
            else:
                show_caption = not caption_printed
            render_detailed_signal_table(asset, title, sigs, px, confidence_level=args.ci, used_student_t_mapping=args.t_map, show_caption=show_caption)
            caption_printed = caption_printed or show_caption
            
            # Show augmentation layer summary if any are active (first signal)
            if sigs:
                render_augmentation_layers_summary(sigs[0])
            
            explanations = []

        # Display diagnostics if computed (Level-7: model falsifiability)
        if diagnostics and (args.diagnostics or args.diagnostics_lite):
            from rich.table import Table
            console = Console()
            diag_table = Table(title=f"📊 Diagnostics for {asset} — Model Falsifiability Metrics")
            diag_table.add_column("Metric", justify="left", style="cyan")
            diag_table.add_column("Value", justify="right")

            # Log-likelihood monitoring
            if "garch_log_likelihood" in diagnostics:
                diag_table.add_row("GARCH(1,1) Log-Likelihood", f"{diagnostics['garch_log_likelihood']:.2f}")
                diag_table.add_row("GARCH(1,1) AIC", f"{diagnostics['garch_aic']:.2f}")
                diag_table.add_row("GARCH(1,1) BIC", f"{diagnostics['garch_bic']:.2f}")

            # Pillar 1: Kalman filter drift diagnostics (with refinements)
            if "kalman_log_likelihood" in diagnostics:
                diag_table.add_row("Kalman Filter Log-Likelihood", f"{diagnostics['kalman_log_likelihood']:.2f}")
                if "kalman_process_noise_var" in diagnostics:
                    diag_table.add_row("Kalman Process Noise (q)", f"{diagnostics['kalman_process_noise_var']:.6f}")
                if "kalman_n_obs" in diagnostics:
                    diag_table.add_row("Kalman Observations", f"{diagnostics['kalman_n_obs']}")

                # Refinement 1: q optimization results
                if "kalman_q_optimal" in diagnostics and "kalman_q_heuristic" in diagnostics:
                    q_opt = diagnostics["kalman_q_optimal"]
                    q_heur = diagnostics["kalman_q_heuristic"]
                    q_optimized = diagnostics.get("kalman_q_optimization_attempted", False)
                    if np.isfinite(q_opt) and np.isfinite(q_heur):
                        ratio = q_opt / q_heur if q_heur > 0 else 1.0
                        opt_label = "optimized" if q_optimized else "heuristic"
                        diag_table.add_row(f"  q ({opt_label})", f"{q_opt:.6f} ({ratio:.2f}× heuristic)")

                # Refinement 2: Kalman gain (situational awareness)
                if "kalman_gain_mean" in diagnostics and "kalman_gain_recent" in diagnostics:
                    gain_mean = diagnostics["kalman_gain_mean"]
                    gain_recent = diagnostics["kalman_gain_recent"]
                    if np.isfinite(gain_mean):
                        # Interpretation: high gain = aggressive learning, low gain = stable drift
                        interpretation = "aggressive" if gain_mean > 0.3 else ("moderate" if gain_mean > 0.1 else "stable")
                        diag_table.add_row(f"  Kalman Gain (mean)", f"{gain_mean:.4f} [{interpretation}]")
                    if np.isfinite(gain_recent):
                        diag_table.add_row(f"  Kalman Gain (recent)", f"{gain_recent:.4f}")

                # Refinement 3: Innovation whiteness (model adequacy)
                if "innovation_ljung_box_pvalue" in diagnostics:
                    pvalue = diagnostics["innovation_ljung_box_pvalue"]
                    model_adequate = diagnostics.get("innovation_model_adequate", None)
                    lags = diagnostics.get("innovation_lags_tested", 0)
                    if np.isfinite(pvalue) and model_adequate is not None:
                        color = "green" if model_adequate else "red"
                        status = "PASS" if model_adequate else "FAIL"
                        diag_table.add_row(f"  Innovation Whiteness (Ljung-Box)", f"[{color}]{status}[/{color}] (p={pvalue:.3f}, lags={lags})")

                # Level-7 Refinement: Heteroskedastic process noise (q_t = c * σ_t²)
                if "kalman_heteroskedastic_mode" in diagnostics:
                    hetero_mode = diagnostics.get("kalman_heteroskedastic_mode", False)
                    c_opt = diagnostics.get("kalman_c_optimal")
                    if hetero_mode and c_opt is not None and np.isfinite(c_opt):
                        diag_table.add_row(f"  Process Noise Mode", f"[cyan]Heteroskedastic[/cyan] (q_t = c·σ_t²)")
                        diag_table.add_row(f"  Scaling Factor (c)", f"{c_opt:.6f}")
                        # Show q_t statistics if available
                        q_t_mean = diagnostics.get("kalman_q_t_mean")
                        q_t_std = diagnostics.get("kalman_q_t_std")
                        q_t_min = diagnostics.get("kalman_q_t_min")
                        q_t_max = diagnostics.get("kalman_q_t_max")
                        if q_t_mean is not None and np.isfinite(q_t_mean):
                            diag_table.add_row(f"  q_t (mean ± std)", f"{q_t_mean:.6f} ± {q_t_std:.6f}" if q_t_std and np.isfinite(q_t_std) else f"{q_t_mean:.6f}")
                        if q_t_min is not None and q_t_max is not None and np.isfinite(q_t_min) and np.isfinite(q_t_max):
                            diag_table.add_row(f"  q_t range [min, max]", f"[{q_t_min:.6f}, {q_t_max:.6f}]")
                    elif not hetero_mode:
                        diag_table.add_row(f"  Process Noise Mode", f"Homoskedastic (constant q)")

                # Level-7+ Refinement: Robust Kalman filtering with Student-t innovations
                if "kalman_robust_t_mode" in diagnostics:
                    robust_t = diagnostics.get("kalman_robust_t_mode", False)
                    nu_robust = diagnostics.get("kalman_nu_robust")
                    if robust_t and nu_robust is not None and np.isfinite(nu_robust):
                        diag_table.add_row(f"  Innovation Distribution", f"[magenta]Student-t[/magenta] (robust filtering)")
                        diag_table.add_row(f"  Innovation ν (degrees of freedom)", f"{nu_robust:.2f}")
                    elif not robust_t:
                        diag_table.add_row(f"  Innovation Distribution", f"Gaussian (standard)")

                # Level-7+ Refinement: Regime-dependent drift priors
                if "kalman_regime_prior_used" in diagnostics:
                    regime_prior_used = diagnostics.get("kalman_regime_prior_used", False)
                    if regime_prior_used:
                        regime_current = diagnostics.get("kalman_regime_current", "")
                        drift_prior = diagnostics.get("kalman_regime_drift_prior")
                        if regime_current and drift_prior is not None and np.isfinite(drift_prior):
                            diag_table.add_row(f"  Drift Prior (regime-aware)", f"[yellow]Enabled[/yellow] (regime: {regime_current})")
                            diag_table.add_row(f"  E[μ | Regime={regime_current}]", f"{drift_prior:+.6f}")
                    else:
                        diag_table.add_row(f"  Drift Prior", f"Neutral (μ₀ = 0)")

            if "hmm_log_likelihood" in diagnostics:
                diag_table.add_row("HMM Regime Log-Likelihood", f"{diagnostics['hmm_log_likelihood']:.2f}")
                diag_table.add_row("HMM AIC", f"{diagnostics['hmm_aic']:.2f}")
                diag_table.add_row("HMM BIC", f"{diagnostics['hmm_bic']:.2f}")

            if "student_t_log_likelihood" in diagnostics:
                diag_table.add_row("Student-t Tail Log-Likelihood", f"{diagnostics['student_t_log_likelihood']:.2f}")
                diag_table.add_row("Student-t Degrees of Freedom (ν)", f"{diagnostics['student_t_nu']:.2f}")

                # Tier 2: Display ν standard error (posterior parameter variance)
                if "student_t_se_nu" in diagnostics:
                    se_nu = diagnostics["student_t_se_nu"]
                    if np.isfinite(se_nu) and se_nu > 0:
                        nu_hat = diagnostics.get("student_t_nu", float("nan"))
                        # Coefficient of variation: SE/estimate (relative uncertainty)
                        cv_nu = (se_nu / nu_hat) if np.isfinite(nu_hat) and nu_hat > 0 else float("nan")
                        uncertainty_level = "low" if cv_nu < 0.05 else ("moderate" if cv_nu < 0.10 else "high")
                        diag_table.add_row("  SE(ν) [posterior uncertainty]", f"{se_nu:.3f} ({cv_nu*100:.1f}% CV, {uncertainty_level})")
                    else:
                        diag_table.add_row("  SE(ν) [posterior uncertainty]", f"{se_nu:.3f}")

            # Tier 2: Parameter Uncertainty Summary (μ, σ, ν)
            param_unc_env = os.getenv("PARAM_UNC", "sample").strip().lower()
            nu_sample_env = os.getenv("NU_SAMPLE", "true").strip().lower()

            param_unc_active = {
                "μ (drift)": "Kalman var_kf → process noise q",
                "σ (volatility)": f"GARCH sampling: {'✓ enabled' if param_unc_env == 'sample' else '✗ disabled'}",
                "ν (tails)": f"Student-t sampling: {'✓ enabled' if nu_sample_env == 'true' else '✗ disabled'}"
            }

            diag_table.add_row("", "")  # spacer
            diag_table.add_row("[bold cyan]Tier 2: Posterior Parameter Variance[/bold cyan]", "[bold]Status[/bold]")
            for param, status in param_unc_active.items():
                if "✓" in status:
                    diag_table.add_row(f"  {param}", f"[#00d700]{status}[/#00d700]")
                elif "✗" in status:
                    diag_table.add_row(f"  {param}", f"[yellow]{status}[/yellow]")
                else:
                    diag_table.add_row(f"  {param}", status)

            # Parameter stability (recent drift z-scores)
            drift_cols = [k for k in diagnostics.keys() if k.startswith("recent_") and k.endswith("_drift_z")]
            if drift_cols:
                diag_table.add_row("", "")  # spacer
                diag_table.add_row("[bold]Parameter Stability[/bold]", "[bold]Recent Drift (z-score)[/bold]")
                for col in drift_cols:
                    param_name = col.replace("recent_", "").replace("_drift_z", "")
                    val = diagnostics[col]
                    if np.isfinite(val):
                        color = "green" if abs(val) < 2.0 else ("yellow" if abs(val) < 3.0 else "red")
                        diag_table.add_row(f"  {param_name}", f"[{color}]{val:+.2f}[/{color}]")

            # Out-of-sample test results (if enabled)
            oos_keys = [k for k in diagnostics.keys() if k.startswith("oos_") and k.endswith("_hit_rate")]
            if oos_keys:
                diag_table.add_row("", "")  # spacer
                diag_table.add_row("[bold]Out-of-Sample Tests[/bold]", "[bold]Direction Hit Rate[/bold]")
                for key in oos_keys:
                    horizon_label = key.replace("oos_", "").replace("_hit_rate", "")
                    hit_rate = diagnostics[key]
                    if np.isfinite(hit_rate):
                        color = "green" if hit_rate >= 0.55 else ("yellow" if hit_rate >= 0.50 else "red")
                        diag_table.add_row(f"  {horizon_label}", f"[{color}]{hit_rate*100:.1f}%[/{color}]")

            console.print(diag_table)
            console.print("")  # blank line

            # Display PIT calibration report if available
            if "pit_calibration" in diagnostics:
                try:
                    from calibration.pit_calibration import format_calibration_report
                    calibration_report = format_calibration_report(
                        calibration_results=diagnostics["pit_calibration"],
                        asset_name=asset
                    )
                    console.print(calibration_report)
                except Exception:
                    pass

            # Display model comparison results if available
            if "model_comparison" in diagnostics and diagnostics["model_comparison"]:
                from rich.table import Table
                comparison_results = diagnostics["model_comparison"]

                # Create comparison table for each category
                for category, result in comparison_results.items():
                    if result is None or not hasattr(result, 'winner_aic'):
                        continue

                    category_title = {
                        'volatility': 'Volatility Models',
                        'tails': 'Tail Distribution Models',
                        'drift': 'Drift Models'
                    }.get(category, category.title())

                    comp_table = Table(title=f"📊 Model Comparison: {category_title} — {asset}")
                    comp_table.add_column("Model", justify="left", style="cyan")
                    comp_table.add_column("Params", justify="right")
                    comp_table.add_column("Log-Likelihood", justify="right")
                    comp_table.add_column("AIC", justify="right")
                    comp_table.add_column("BIC", justify="right")
                    comp_table.add_column("Δ AIC", justify="right")
                    comp_table.add_column("Δ BIC", justify="right")
                    comp_table.add_column("Akaike Wt", justify="right")

                    for model in result.models:
                        name = model.name

                        # Highlight winners
                        if name == result.winner_aic and name == result.winner_bic:
                            name = f"[bold #00d700]{name}[/bold #00d700] ⭐"
                        elif name == result.winner_aic:
                            name = f"[bold yellow]{name}[/bold yellow] (AIC)"
                        elif name == result.winner_bic:
                            name = f"[bold blue]{name}[/bold blue] (BIC)"

                        delta_aic = result.delta_aic.get(model.name, float('nan'))
                        delta_bic = result.delta_bic.get(model.name, float('nan'))
                        weight = result.akaike_weights.get(model.name, 0.0)

                        # Color code deltas (lower is better)
                        if np.isfinite(delta_aic):
                            if delta_aic < 2.0:
                                delta_aic_str = f"[#00d700]{delta_aic:+.1f}[/#00d700]"
                            elif delta_aic < 7.0:
                                delta_aic_str = f"[yellow]{delta_aic:+.1f}[/yellow]"
                            else:
                                delta_aic_str = f"[indian_red1]{delta_aic:+.1f}[/indian_red1]"
                        else:
                            delta_aic_str = "—"

                        if np.isfinite(delta_bic):
                            if delta_bic < 2.0:
                                delta_bic_str = f"[#00d700]{delta_bic:+.1f}[/#00d700]"
                            elif delta_bic < 10.0:
                                delta_bic_str = f"[yellow]{delta_bic:+.1f}[/yellow]"
                            else:
                                delta_bic_str = f"[indian_red1]{delta_bic:+.1f}[/indian_red1]"
                        else:
                            delta_bic_str = "—"

                        comp_table.add_row(
                            name,
                            str(model.n_params),
                            f"{model.log_likelihood:.2f}",
                            f"{model.aic:.2f}",
                            f"{model.bic:.2f}",
                            delta_aic_str,
                            delta_bic_str,
                            f"{weight:.1%}" if np.isfinite(weight) else "—"
                        )

                    # Add recommendation row
                    comp_table.add_row("", "", "", "", "", "", "", "")
                    comp_table.add_row(
                        f"[bold]Recommendation:[/bold]",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        ""
                    )

                    console.print(comp_table)
                    console.print(f"[dim]{result.recommendation}[/dim]\n")

                # Summary interpretation
                console.print("[bold cyan]Model Comparison Interpretation:[/bold cyan]")
                console.print("[dim]• Δ AIC/BIC < 2: Substantial support (competitive models)[/dim]")
                console.print("[dim]• Δ AIC/BIC 4-7: Considerably less support[/dim]")
                console.print("[dim]• Δ AIC/BIC > 10: Essentially no support[/dim]")
                console.print("[dim]• Akaike weight: Probability this model is best[/dim]")
                console.print("[dim]• Lower AIC/BIC = better (fit + parsimony tradeoff)[/dim]\n")

        # 🧪 Level-7 Validation Science: Kalman Filter Validation Suite
        if args.validate_kalman:
            try:
                from kalman_validation import (
                    run_full_validation_suite,
                    validate_drift_reasonableness,
                    compare_predictive_likelihood,
                    validate_pit_calibration,
                    analyze_stress_regime_behavior
                )
                from rich.table import Table
                from rich.panel import Panel

                console = Console()
                console.print("\n")
                console.print(Panel.fit(
                    f"🧪 [bold cyan]Level-7 Validation Science[/bold cyan] — {asset}\n"
                    "[dim]Does my model behave like reality?[/dim]",
                    border_style="cyan"
                ))

                # Extract required series from features
                ret = feats.get("ret")
                mu_kf = feats.get("mu_kf", feats.get("mu"))
                var_kf = feats.get("var_kf", pd.Series(0.0, index=ret.index))
                vol = feats.get("vol")

                if mu_kf is not None and ret is not None and vol is not None:
                    # Prepare plot directory if plots requested
                    plot_dir = None
                    if args.validation_plots:
                        plot_dir = "src/data/plots/kalman_validation"
                        os.makedirs(plot_dir, exist_ok=True)

                    # 1. Drift Reasonableness Validation
                    console.print("\n[bold yellow]1. Posterior Drift Reasonableness[/bold yellow]")
                    drift_result = validate_drift_reasonableness(
                        px, ret, mu_kf, var_kf, asset_name=asset,
                        plot=args.validation_plots,
                        save_path=f"{plot_dir}/{asset}_drift_validation.png" if plot_dir else None
                    )

                    drift_table = Table(title="Drift Sanity Checks", show_header=True)
                    drift_table.add_column("Metric", style="cyan")
                    drift_table.add_column("Value", justify="right")
                    drift_table.add_column("Status", justify="center")

                    drift_table.add_row(
                        "Observations",
                        str(drift_result.observations),
                        ""
                    )
                    drift_table.add_row(
                        "Drift Smoothness Ratio",
                        f"{drift_result.drift_smoothness_ratio:.4f}",
                        "✅" if drift_result.drift_smoothness_ratio < 0.5 else "⚠️"
                    )
                    drift_table.add_row(
                        "Crisis Uncertainty Spike",
                        f"{drift_result.crisis_uncertainty_spike:.2f}×",
                        "✅" if drift_result.crisis_uncertainty_spike > 1.5 else "⚠️"
                    )
                    drift_table.add_row(
                        "Regime Breaks Detected",
                        "Yes" if drift_result.regime_break_detected else "No",
                        "✅" if drift_result.regime_break_detected else "ℹ️"
                    )
                    drift_table.add_row(
                        "Noise Tracking Score",
                        f"{drift_result.noise_tracking_score:.4f}",
                        "✅" if drift_result.noise_tracking_score < 0.4 else "⚠️"
                    )

                    console.print(drift_table)
                    console.print(f"[dim]{drift_result.diagnostic_message}[/dim]\n")

                    # 2. Predictive Likelihood Improvement
                    console.print("[bold yellow]2. Predictive Likelihood Improvement[/bold yellow]")
                    ll_result = compare_predictive_likelihood(px, asset_name=asset)

                    ll_table = Table(title="Model Comparison (Out-of-Sample)", show_header=True)
                    ll_table.add_column("Model", style="cyan")
                    ll_table.add_column("Log-Likelihood", justify="right")
                    ll_table.add_column("Δ LL", justify="right")

                    ll_table.add_row("Kalman Filter", f"{ll_result.ll_kalman:.2f}", "—")
                    ll_table.add_row(
                        "Zero Drift (μ=0)",
                        f"{ll_result.ll_zero_drift:.2f}",
                        f"[#00d700]{ll_result.delta_ll_vs_zero:+.2f}[/#00d700]" if ll_result.delta_ll_vs_zero > 0 else f"[indian_red1]{ll_result.delta_ll_vs_zero:+.2f}[/indian_red1]"
                    )
                    ll_table.add_row(
                        "EWMA Drift",
                        f"{ll_result.ll_ewma_drift:.2f}",
                        f"[#00d700]{ll_result.delta_ll_vs_ewma:+.2f}[/#00d700]" if ll_result.delta_ll_vs_ewma > 0 else f"[indian_red1]{ll_result.delta_ll_vs_ewma:+.2f}[/indian_red1]"
                    )
                    ll_table.add_row(
                        "Constant Drift",
                        f"{ll_result.ll_constant_drift:.2f}",
                        f"[#00d700]{ll_result.delta_ll_vs_constant:+.2f}[/#00d700]" if ll_result.delta_ll_vs_constant > 0 else f"[indian_red1]{ll_result.delta_ll_vs_constant:+.2f}[/indian_red1]"
                    )

                    console.print(ll_table)
                    console.print(f"[bold]Best Model:[/bold] {ll_result.best_model}")
                    console.print(f"[dim]{ll_result.diagnostic_message}[/dim]\n")

                    # 3. PIT Calibration Check
                    console.print("[bold yellow]3. Probability Integral Transform (PIT) Calibration[/bold yellow]")
                    pit_result = validate_pit_calibration(
                        px, ret, mu_kf, var_kf, vol, asset_name=asset,
                        plot=args.validation_plots,
                        save_path=f"{plot_dir}/{asset}_pit_calibration.png" if plot_dir else None
                    )

                    pit_table = Table(title="Forecast Calibration", show_header=True)
                    pit_table.add_column("Metric", style="cyan")
                    pit_table.add_column("Value", justify="right")
                    pit_table.add_column("Expected", justify="right")
                    pit_table.add_column("Status", justify="center")

                    pit_table.add_row(
                        "Observations",
                        str(pit_result.n_observations),
                        "—",
                        ""
                    )
                    pit_table.add_row(
                        "KS Statistic",
                        f"{pit_result.ks_statistic:.4f}",
                        "—",
                        ""
                    )
                    pit_table.add_row(
                        "KS p-value",
                        f"{pit_result.ks_pvalue:.4f}",
                        "> 0.05",
                        "✅" if pit_result.ks_pvalue >= 0.05 else "⚠️"
                    )
                    pit_table.add_row(
                        "PIT Mean",
                        f"{pit_result.pit_mean:.4f}",
                        "0.5000",
                        "✅" if abs(pit_result.pit_mean - 0.5) < 0.05 else "⚠️"
                    )
                    pit_table.add_row(
                        "PIT Std Dev",
                        f"{pit_result.pit_std:.4f}",
                        f"{expected_std:.4f}",
                        "✅" if abs(pit_result.pit_std - expected_std) < 0.05 else "⚠️"
                    )

                    console.print(pit_table)
                    console.print(f"[dim]{pit_result.diagnostic_message}[/dim]\n")

                    # 4. Stress-Regime Behavior
                    console.print("[bold yellow]4. Stress-Regime Behavior Analysis[/bold yellow]")
                    stress_result = analyze_stress_regime_behavior(
                        px, ret, mu_kf, var_kf, vol, asset_name=asset
                    )

                    stress_table = Table(title="Risk Intelligence", show_header=True)
                    stress_table.add_column("Metric", style="cyan")
                    stress_table.add_column("Normal", justify="right")
                    stress_table.add_column("Stress", justify="right")
                    stress_table.add_column("Ratio", justify="right")

                    stress_table.add_row(
                        "Drift Uncertainty σ(μ̂)",
                        f"{stress_result.avg_uncertainty_normal:.6f}",
                        f"{stress_result.avg_uncertainty_stress:.6f}",
                        f"[#00d700]{stress_result.uncertainty_spike_ratio:.2f}×[/#00d700]" if stress_result.uncertainty_spike_ratio > 1.2 else f"{stress_result.uncertainty_spike_ratio:.2f}×"
                    )
                    stress_table.add_row(
                        "Position Size (EU)",
                        f"{stress_result.avg_kelly_normal:.4f}",
                        f"{stress_result.avg_kelly_stress:.4f}",
                        f"[#00d700]{stress_result.kelly_reduction_ratio:.2f}×[/#00d700]" if stress_result.kelly_reduction_ratio < 0.9 else f"{stress_result.kelly_reduction_ratio:.2f}×"
                    )

                    console.print(stress_table)

                    if stress_result.stress_periods_detected:
                        console.print(f"\n[bold]Stress Periods Detected:[/bold] {len(stress_result.stress_periods_detected)}")
                        for i, (start, end) in enumerate(stress_result.stress_periods_detected[:5], 1):
                            console.print(f"  {i}. {start} → {end}")
                        if len(stress_result.stress_periods_detected) > 5:
                            console.print(f"  ... and {len(stress_result.stress_periods_detected) - 5} more")

                    console.print(f"\n[dim]{stress_result.diagnostic_message}[/dim]\n")

                    # Overall validation summary
                    all_passed = (
                        drift_result.validation_passed and
                        ll_result.improvement_significant and
                        pit_result.calibration_passed and
                        stress_result.system_backed_off
                    )

                    if all_passed:
                        console.print(Panel.fit(
                            "[bold #00d700]✅ ALL VALIDATION CHECKS PASSED[/bold #00d700]\n"
                            "[dim]Model demonstrates structural realism and statistical rigor.[/dim]",
                            border_style="green"
                        ))
                    else:
                        console.print(Panel.fit(
                            "[bold yellow]⚠️ SOME VALIDATION CHECKS FAILED[/bold yellow]\n"
                            "[dim]Review diagnostics above for tuning guidance.[/dim]",
                            border_style="yellow"
                        ))
                else:
                    console.print("[indian_red1]⚠️ Kalman filter data not available for validation[/indian_red1]")

            except Exception as e:
                console.print(f"[indian_red1]⚠️ Validation failed: {e}[/indian_red1]")
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")

        # Build summary row for this asset
        asset_label = build_asset_display_label(asset, title)
        horizon_signals = {
            int(s.horizon_days): {
                "label": s.label,
                "profit_pln": float(s.profit_pln),
                "ue_up": float(s.ue_up),
                "ue_down": float(s.ue_down),
                "p_up": float(s.p_up),
                "exp_ret": float(s.exp_ret),
            }
            for s in sigs
        }
        nearest_label = sigs[0].label if sigs else "HOLD"
        
        # Compute crash risk for this asset (uses momentum-based multi-factor model)
        crash_risk_score = 0
        if ASSET_CRASH_RISK_AVAILABLE and px is not None and len(px) > 50:
            try:
                # Get volume if available in features
                volume_series = feats.get("volume") if feats else None
                crash_result = compute_asset_crash_risk(px, volume_series, canon)
                crash_risk_score = crash_result.crash_risk_score
                # Cache for potential reuse
                cache_crash_risk(canon, crash_result)
            except Exception:
                crash_risk_score = 0
        
        # Compute momentum score for this asset (-100 to +100)
        momentum_score = compute_momentum_score(px, feats)
        
        summary_rows.append({
            "asset_label": asset_label,
            "horizon_signals": horizon_signals,
            "nearest_label": nearest_label,
            "sector": get_sector(canon),
            "crash_risk_score": crash_risk_score,
            "momentum_score": momentum_score,
            "conviction": enrichment.get("conviction"),
            "kelly": enrichment.get("kelly"),
            "signal_ttl": enrichment.get("signal_ttl"),
        })

        # Track regime and model for end-of-run summary
        try:
            import re
            
            # Get model name from feats - multiple sources
            model_name = "Unknown"
            if feats:
                # Try kalman_metadata first
                kalman_meta = feats.get("kalman_metadata", {})
                if isinstance(kalman_meta, dict):
                    # best_model has the full name including _momentum suffix
                    model_name = kalman_meta.get("best_model", "")
                    if not model_name:
                        model_name = kalman_meta.get("kalman_noise_model", "")
                    if not model_name:
                        model_name = kalman_meta.get("model_selected", "")
                
                # Try direct feats keys if not found
                if not model_name or model_name == "Unknown":
                    model_name = feats.get("best_model", "")
                if not model_name or model_name == "Unknown":
                    model_name = feats.get("kalman_noise_model", "")
                
                # Default to gaussian if still empty
                if not model_name:
                    model_name = "gaussian"
            
            # Get regime from feats - prioritize HMM result which is the most reliable
            regime_name = "Unknown"
            if feats:
                # PRIMARY: Try HMM result for regime (most reliable - uses 3-state Gaussian HMM)
                hmm_result = feats.get("hmm_result")
                if isinstance(hmm_result, dict) and hmm_result:
                    # fit_hmm_regimes returns regime_series (pd.Series of regime labels)
                    # Get current regime from the last value in the series
                    regime_series = hmm_result.get("regime_series")
                    if regime_series is not None and hasattr(regime_series, 'iloc') and len(regime_series) > 0:
                        regime_name = str(regime_series.iloc[-1])
                    else:
                        # Fallback: check if there's a current_regime key (older format)
                        regime_name = hmm_result.get("current_regime", "")
                
                # FALLBACK: Try kalman_metadata for regime info
                if not regime_name or regime_name == "Unknown":
                    kalman_meta = feats.get("kalman_metadata", {})
                    if isinstance(kalman_meta, dict):
                        regime_info = kalman_meta.get("kalman_regime_info", {})
                        if isinstance(regime_info, dict) and regime_info:
                            regime_name = regime_info.get("regime_name", "")
                            if not regime_name:
                                regime_idx = regime_info.get("regime_index")
                                regime_map = {0: "LOW_VOL_TREND", 1: "HIGH_VOL_TREND", 2: "LOW_VOL_RANGE", 3: "HIGH_VOL_RANGE", 4: "CRISIS_JUMP"}
                                regime_name = regime_map.get(regime_idx, "")
                
                # FALLBACK: Try regime_params directly
                if not regime_name or regime_name == "Unknown":
                    regime_info = feats.get("regime_params", {})
                    if isinstance(regime_info, dict) and regime_info:
                        # Get first regime key as current
                        regime_name = next(iter(regime_info.keys()), "Unknown")
                
                # FALLBACK: Volatility-based regime classification (for assets without HMM)
                # This handles new/short-history assets where HMM returns None
                if not regime_name or regime_name == "Unknown":
                    vol_series = feats.get("vol") if feats else None
                    if vol_series is not None and hasattr(vol_series, 'iloc') and len(vol_series) > 0:
                        try:
                            current_vol = float(vol_series.iloc[-1])
                            # Annualized volatility thresholds (approximate)
                            # calm: < 15%, trending: 15-30%, crisis: > 30%
                            if current_vol < 0.15:
                                regime_name = "calm"
                            elif current_vol < 0.30:
                                regime_name = "trending"
                            else:
                                regime_name = "crisis"
                        except (ValueError, TypeError):
                            pass
                
                # Final default if still empty
                if not regime_name:
                    regime_name = "Unknown"
            
            # Convert model name to short display name
            model_short = model_name
            is_momentum = '_momentum' in model_name
            base_name = model_name.replace('_momentum', '')
            
            # Use the same model info lookup as the Model Weights display
            model_info_lookup = {
                'kalman_gaussian': 'Gaussian',
                'kalman_phi_gaussian': 'φ-Gaussian',
                'kalman_gaussian_unified': 'Gaussian-Uni [U]',
                'kalman_phi_gaussian_unified': 'φ-Gaussian-Uni [U]',
                'kalman_gaussian_momentum': 'Gaussian+Momentum',
                'kalman_phi_gaussian_momentum': 'φ-Gaussian+Momentum',
                'gaussian': 'Gaussian',
            }
            if model_name in model_info_lookup:
                model_short = model_info_lookup[model_name]
            # Handle VoV models with gamma extraction
            elif 'vov' in base_name.lower() and 'student_t' in base_name.lower():
                gamma_match = re.search(r'_g(\d+\.?\d*)', base_name)
                nu_match = re.search(r'nu_(\d+)', base_name)
                if gamma_match and nu_match:
                    model_short = f'φ-T(ν={nu_match.group(1)},γ={gamma_match.group(1)})'
                elif gamma_match:
                    model_short = f'φ-T(VoV,γ={gamma_match.group(1)})'
                elif nu_match:
                    model_short = f'φ-T(ν={nu_match.group(1)},VoV)'
                else:
                    model_short = 'φ-T(VoV)'
                if is_momentum:
                    model_short += '+Momentum'
            # Handle Two-Piece models
            elif 'nul' in base_name.lower() and 'nur' in base_name.lower():
                nul_match = re.search(r'nul(\d+)', base_name.lower())
                nur_match = re.search(r'nur(\d+)', base_name.lower())
                if nul_match and nur_match:
                    model_short = f'φ-T(νL={nul_match.group(1)},νR={nur_match.group(1)})'
                else:
                    model_short = 'φ-T(2P)'
                if is_momentum:
                    model_short += '+Momentum'
            # Handle Mixture models
            elif 'mix' in base_name.lower() and 'student_t' in base_name.lower():
                mix_match = re.search(r'mix_(\d+)_(\d+)', base_name)
                if mix_match:
                    model_short = f'φ-T(Mix:{mix_match.group(1)}/{mix_match.group(2)})'
                else:
                    model_short = 'φ-T(Mix)'
                if is_momentum:
                    model_short += '+Momentum'
            elif 'phi_student_t_nu_' in model_name:
                # Extract nu value
                nu_match = re.search(r'nu_(\d+)', model_name)
                if nu_match:
                    nu_val = nu_match.group(1)
                    model_short = f'φ-T(ν={nu_val})'
                    if is_momentum:
                        model_short += '+Momentum'
            elif is_momentum:
                model_short = base_name.replace('_', '-') + '+Momentum'
            
            # Update tracker
            if regime_name not in regime_model_tracker:
                regime_model_tracker[regime_name] = {}
            if model_short not in regime_model_tracker[regime_name]:
                regime_model_tracker[regime_name][model_short] = 0
            regime_model_tracker[regime_name][model_short] += 1
        except Exception:
            pass  # Silent fail for tracking

        # Prepare JSON block
        block = {
            "symbol": asset,
            "title": title,
            "as_of": str(px.index[-1].date()),
            "last_close": last_close,
            "notional_pln": NOTIONAL_PLN,
            "signals": [s.__dict__ for s in sigs],
            "ci_level": args.ci,
            "ci_domain": "log_return",
            "profit_ci_domain": "arithmetic_pln",
            "probability_mapping": ("student_t" if args.t_map else "normal"),
            "nu_clip": {"min": 4.5, "max": 500.0},
            "edgeworth_damped": True,
            "kelly_rule": "half",
            "decision_thresholds": thresholds,
            # crash risk score (0-100, momentum-based multi-factor)
            "crash_risk_score": crash_risk_score,
            # volatility modeling metadata
            "vol_source": feats.get("vol_source", "garch11"),
            "garch_params": feats.get("garch_params", {}),
            # tail modeling metadata (global ν)
            "tail_model": "student_t_global",
            "nu_hat": float(feats.get("nu_hat").iloc[-1]) if isinstance(feats.get("nu_hat"), pd.Series) and not feats.get("nu_hat").empty else 50.0,
            "nu_bounds": {"min": 4.5, "max": 500.0},
            "nu_info": feats.get("nu_info", {}),
            # stochastic volatility metadata (Level-7: full posterior uncertainty)
            "stochastic_volatility": {
                "enabled": True,
                "method": "bayesian_garch_sampling",
                "parameter_sampling": os.getenv("PARAM_UNC", "sample"),
                "uncertainty_propagated": True,
                "volatility_ci_tracked": True,
                "description": "Volatility treated as latent stochastic process with posterior uncertainty. GARCH parameters sampled from N(theta_hat, Cov) per path. Full h_t trajectories tracked and volatility credible intervals reported per horizon."
            },
        }

        # Add diagnostics to JSON if computed (Level-7 falsifiability)
        if diagnostics:
            # Filter out non-serializable objects (DataFrames) for JSON
            serializable_diagnostics = {}
            for k, v in diagnostics.items():
                if k in ("parameter_stability", "out_of_sample"):
                    # Skip raw DataFrames; summary metrics already in top-level diagnostics
                    continue
                if isinstance(v, (int, float, str, bool, type(None))):
                    serializable_diagnostics[k] = v
                elif isinstance(v, dict):
                    # Nested dicts are OK if they contain serializable values
                    serializable_diagnostics[k] = v
            block["diagnostics"] = serializable_diagnostics

        # Add Epic 8 enrichment data (conviction, Kelly, TTL) to JSON export
        if enrichment:
            block["enrichment"] = enrichment

        all_blocks.append(block)

        # Prepare CSV rows
        if args.simple:
            for i, s in enumerate(sigs):
                csv_rows_simple.append({
                    "asset": title,
                    "symbol": asset,
                    "timeframe": format_horizon_label(s.horizon_days),
                    "chance_up_pct": f"{s.p_up*100:.1f}",
                    "recommendation": s.label,
                    "why": explanations[i],
                })
        else:
            for s in sigs:
                row = s.__dict__.copy()
                row.update({
                    "asset": title,
                    "symbol": asset,
                })
                csv_rows_detailed.append(row)

    # After processing all assets, print a compact summary
    try:
        # Group summary rows by sector and render one table per sector
        render_sector_summary_tables(summary_rows, horizons)
        # Add high-conviction signals summary for short-term trading
        render_strong_signals_summary(summary_rows, horizons=[1, 3, 7])
        
        # Save high conviction signals to src/data/high_conviction/ with options data
        if HIGH_CONVICTION_STORAGE_AVAILABLE:
            try:
                result = save_high_conviction_signals(
                    summary_rows, 
                    horizons=[1, 3, 7],
                    fetch_options=True,
                    fetch_prices=True,
                )
                if result.get("buy", 0) > 0 or result.get("sell", 0) > 0:
                    Console().print(
                        f"\n[dim]💾 Saved {result['buy']} buy + {result['sell']} sell signals "
                        f"to src/data/high_conviction/[/dim]"
                    )
                    # Generate candlestick charts for strong signals
                    if SIGNAL_CHARTS_AVAILABLE:
                        try:
                            generate_signal_charts(quiet=False)
                        except Exception as chart_e:
                            if os.getenv("DEBUG"):
                                Console().print(f"[dim]Signal chart error: {chart_e}[/dim]")
            except Exception as hc_e:
                if os.getenv("DEBUG"):
                    Console().print(f"[dim]High conviction storage error: {hc_e}[/dim]")
        
        # Below SMA50 analysis — table + charts
        if SIGNAL_CHARTS_AVAILABLE and find_below_sma50_stocks is not None:
            try:
                below_sma50 = find_below_sma50_stocks(summary_rows)
                if below_sma50:
                    render_below_sma50_table(below_sma50, quiet=False)
                    generate_sma_charts(below_sma50, quiet=False)
            except Exception as sma_e:
                if os.getenv("DEBUG"):
                    Console().print(f"[dim]SMA50 analysis error: {sma_e}[/dim]")
        
        # Index fund / ETF charts
        if SIGNAL_CHARTS_AVAILABLE and generate_index_charts is not None:
            try:
                generate_index_charts(quiet=False)
            except Exception as idx_e:
                if os.getenv("DEBUG"):
                    Console().print(f"[dim]Index chart error: {idx_e}[/dim]")
        
        # Show unified risk dashboard (replaces fragmented risk temperature display)
        # February 2026: Full risk dashboard matching `make risk` output
        if UNIFIED_RISK_DASHBOARD_AVAILABLE:
            try:
                compute_and_render_unified_risk(
                    start_date="2020-01-01",
                    suppress_output=True,
                    console=Console(),
                    use_parallel=True,  # Use maximum processors for speed
                )
            except Exception as rd_e:
                if os.getenv("DEBUG"):
                    Console().print(f"[dim]Unified risk dashboard error: {rd_e}[/dim]")
                # Fallback to legacy risk temperature summary
                if RISK_TEMPERATURE_AVAILABLE:
                    try:
                        risk_temp_result = get_cached_risk_temperature(
                            start_date="2020-01-01",
                            notional=NOTIONAL_PLN,
                            estimated_gap_risk=0.03,
                        )
                        render_risk_temperature_summary(risk_temp_result)
                    except Exception:
                        pass
        elif RISK_TEMPERATURE_AVAILABLE:
            try:
                risk_temp_result = get_cached_risk_temperature(
                    start_date="2020-01-01",
                    notional=NOTIONAL_PLN,
                    estimated_gap_risk=0.03,
                )
                render_risk_temperature_summary(risk_temp_result)
            except Exception as rt_e:
                if os.getenv("DEBUG"):
                    Console().print(f"[dim]Risk temperature display skipped: {rt_e}[/dim]")
    except Exception as e:
        Console().print(f"[yellow]Warning:[/yellow] Could not print summary tables: {e}")

    # Render regime-model summary table (shows which models were selected per regime)
    try:
        if regime_model_tracker:
            render_regime_model_summary(regime_model_tracker, Console(force_terminal=True, width=140))
    except Exception as rms_e:
        if os.getenv("DEBUG"):
            Console().print(f"[dim]Regime-model summary error: {rms_e}[/dim]")

    # =========================================================================
    # EPIC 8 PORTFOLIO-LEVEL ENRICHMENT (April 2026)
    # =========================================================================
    # Pair screening and sector rotation run AFTER all individual assets.
    # Results are added to the payload for JSON export.
    # =========================================================================

    portfolio_enrichment = {}

    # 8.4: Pair Trading — screen for cointegrated pairs across the universe
    if PAIR_TRADING_AVAILABLE and len(success_results) >= 4:
        try:
            price_dict = {}
            for r in success_results:
                if r and r.get("status") == "success" and r.get("px") is not None:
                    sym = r["canon"]
                    price_dict[sym] = r["px"].values
            if len(price_dict) >= 4:
                symbols = list(price_dict.keys())
                pairs = screen_pairs(price_dict, symbols, top_n=10)
                portfolio_enrichment["pairs"] = [
                    {
                        "asset_a": p.asset_a,
                        "asset_b": p.asset_b,
                        "adf_stat": p.adf_stat,
                        "pvalue": p.pvalue,
                        "halflife": p.halflife,
                        "zscore": p.spread_zscore,
                        "signal": p.signal,
                    }
                    for p in pairs
                ]
        except Exception:
            pass

    # 8.5: Sector Rotation — compute sector-level signals
    if SECTOR_ROTATION_AVAILABLE and summary_rows:
        try:
            from decision.sector_rotation import SectorSignal
            sector_forecasts = {}
            for row in summary_rows:
                sector = row.get("sector", "Other")
                h_sigs = row.get("horizon_signals", {})
                # Use 7-day or nearest horizon p_up as forecast proxy
                for h in [7, 3, 1, 30]:
                    if h in h_sigs:
                        p = h_sigs[h].get("p_up", 0.5)
                        sector_forecasts.setdefault(sector, []).append(p - 0.5)
                        break
            if sector_forecasts:
                sector_signals = []
                for sector, forecasts in sector_forecasts.items():
                    avg = sum(forecasts) / len(forecasts) if forecasts else 0
                    breadth = sum(1 for f in forecasts if f > 0) / max(len(forecasts), 1)
                    composite = 0.6 * max(-1, min(1, avg * 10)) + 0.4 * (breadth - 0.5) * 2
                    rec = "OVERWEIGHT" if composite > 0.3 else ("UNDERWEIGHT" if composite < -0.3 else "NEUTRAL")
                    sector_signals.append({
                        "sector": sector,
                        "composite": round(composite, 3),
                        "breadth": round(breadth, 3),
                        "recommendation": rec,
                    })
                sector_signals.sort(key=lambda x: x["composite"], reverse=True)
                portfolio_enrichment["sector_rotation"] = sector_signals
        except Exception:
            pass

    # Build structured failure log for exports
    failure_log = [
        {
            "asset": asset,
            "display_name": info.get("display_name", asset),
            "attempts": info.get("attempts", 0),
            "last_error": info.get("last_error", ""),
            "traceback": info.get("traceback", ""),
        }
        for asset, info in failures.items()
    ]

    # Print failure summary table (if any)
    if failures:
        from rich.table import Table
        fail_table = Table(title="Failed Assets After Retries")
        fail_table.add_column("Asset", style="red", justify="left")
        fail_table.add_column("Display Name", justify="left")
        fail_table.add_column("Attempts", justify="right")
        fail_table.add_column("Last Error", justify="left")
        for asset, info in failures.items():
            fail_table.add_row(asset, str(info.get("display_name", asset)), str(info.get("attempts", "")), str(info.get("last_error", "")))
        Console().print(fail_table)

        # Save failed assets to src/data/failed/ for later purging
        try:
            saved_path = save_failed_assets(failures, append=True)
            Console().print(f"[dim]Failed assets saved to: {saved_path}[/dim]")
            Console().print(f"[dim]Run 'make purge' to purge cached data for failed assets[/dim]")
        except Exception as e:
            Console().print(f"[yellow]Warning:[/yellow] Could not save failed assets: {e}")

    # Exports
    cache_path = args.cache_json or DEFAULT_CACHE_PATH
    payload = {
        "assets": all_blocks,
        "summary_rows": summary_rows,
        "horizons": horizons,
        "column_descriptions": DETAILED_COLUMN_DESCRIPTIONS,
        "simple_column_descriptions": SIMPLIFIED_COLUMN_DESCRIPTIONS,
        "failed_assets": failure_log,
        "portfolio_enrichment": portfolio_enrichment,
    }
    # Validate pipeline output (Epic 30: integration_testing)
    if INTEGRATION_TESTING_AVAILABLE and all_blocks:
        try:
            validation_failures = 0
            for block in all_blocks[:10]:  # Sample first 10 for speed
                asset_name = block.get("asset", block.get("canon", ""))
                tune_params = block.get("tuned_params")
                if tune_params:
                    result = validate_pipeline_output(asset_name, tune_params)
                    if not result.success:
                        validation_failures += 1
            if validation_failures > 0:
                Console().print(f"[yellow]Pipeline validation:[/yellow] {validation_failures} asset(s) with issues")
        except Exception:
            pass
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        Console().print(f"[yellow]Warning:[/yellow] Could not write cache JSON: {e}")

    if args.json:
        try:
            with open(args.json, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            Console().print(f"[yellow]Warning:[/yellow] Could not write JSON export: {e}")

    # Epic 7.4: Persist forecasts to SQLite for historical analysis
    if QUANT_DB_AVAILABLE and all_blocks:
        try:
            db_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data", "quant.db")
            db = QuantDB(db_path)
            from datetime import date as date_type
            today = date_type.today().isoformat()
            rows = []
            for block in all_blocks:
                sym = block.get("symbol", "")
                for sig in block.get("signals", []):
                    rows.append({
                        "symbol": sym,
                        "date": today,
                        "horizon": int(sig.get("horizon_days", 0)),
                        "forecast_pct": float(sig.get("exp_ret", 0) * 100),
                        "p_up": float(sig.get("p_up", 0.5)),
                        "confidence_low": float(sig.get("ci_low", 0)),
                        "confidence_high": float(sig.get("ci_high", 0)),
                        "regime": sig.get("regime", ""),
                        "label": sig.get("label", "HOLD"),
                        "model": sig.get("bma_method", ""),
                    })
            if rows:
                db.insert_forecasts_batch(rows)
            db.close()
        except Exception:
            pass  # DB persistence is best-effort
