#!/usr/bin/env python3
"""
tune_ux.py

World-class UX wrapper for tune.py using the Rich presentation layer.
Provides beautiful, informative output while delegating to the core tuning logic.

Usage:
    python src/tuning/tune_ux.py --dry-run --max-assets 5
    python src/tuning/tune_ux.py --force
    python src/tuning/tune_ux.py
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional
from datetime import datetime

# Suppress verbose tuning output by default to prevent mixing with Rich Live display
# Errors are still captured and displayed prominently
# Use --verbose to see all tuning messages
os.environ['TUNING_QUIET'] = '1'

# Add paths for imports
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import core tuning functionality
from tuning.tune import (
    load_asset_list,
    load_cache,
    save_cache_json,
    _tune_worker,
    _extract_previous_posteriors,
    REGIME_LABELS,
)

# Import Filter Cache for statistics reporting
try:
    from models.filter_cache import (
        get_filter_cache,
        get_cache_stats,
        reset_cache_stats,
        clear_filter_cache,
        FILTER_CACHE_ENABLED,
    )
    FILTER_CACHE_AVAILABLE = True
except ImportError:
    FILTER_CACHE_AVAILABLE = False

# Import PIT-Driven Distribution Escalation
try:
    from calibration.pit_driven_escalation import (
        get_escalation_summary_from_cache,
        extract_escalation_from_result,
        EscalationLevel,
        LEVEL_NAMES,
    )
    PDDE_AVAILABLE = True
except ImportError:
    PDDE_AVAILABLE = False

# Import Control Policy — Authority Boundary Layer (Counter-Proposal v1.0)
try:
    from calibration.control_policy import EscalationStatistics
    CONTROL_POLICY_AVAILABLE = True
except ImportError:
    CONTROL_POLICY_AVAILABLE = False
    EscalationStatistics = None

# Import PIT Penalty — Asymmetric Calibration Governance (February 2026)
try:
    from calibration.pit_penalty import (
        get_pit_critical_stocks,
        PIT_EXIT_THRESHOLD,
        PIT_CRITICAL_THRESHOLDS,
    )
    PIT_PENALTY_AVAILABLE = True
except ImportError:
    PIT_PENALTY_AVAILABLE = False

# Rich imports for presentation layer
import json
import multiprocessing
from collections import Counter
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.rule import Rule
from rich.align import Align

from concurrent.futures import ProcessPoolExecutor, as_completed


# =============================================================================
# TUNING OUTPUT PRESENTATION - Moved from signals_ux.py
# =============================================================================

TUNING_REGIME_LABELS = {
    0: "LOW_VOL_TREND",
    1: "HIGH_VOL_TREND",
    2: "LOW_VOL_RANGE",
    3: "HIGH_VOL_RANGE",
    4: "CRISIS_JUMP",
}

REGIME_COLORS = {
    "LOW_VOL_TREND": "cyan",
    "HIGH_VOL_TREND": "yellow",
    "LOW_VOL_RANGE": "green",
    "HIGH_VOL_RANGE": "orange1",
    "CRISIS_JUMP": "red",
}


def create_tuning_console() -> Console:
    return Console(force_terminal=True, color_system="truecolor", width=140)


def _human_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def render_elite_tuning_summary(
    cache: Dict[str, Dict],
    console: Console = None
) -> None:
    """
    Render Elite Tuning diagnostics summary (v2.0).
    
    This shows aggregate stability metrics from plateau-optimal parameter selection:
    - Average fragility index across assets
    - Distribution of Hessian condition numbers
    - Ridge vs basin detection results
    - Drift vs noise decomposition
    - Assets with fragility warnings
    """
    if console is None:
        console = create_tuning_console()
    
    if not cache:
        return
    
    # Collect elite diagnostics from all assets
    fragility_indices = []
    condition_numbers = []
    fragility_warnings = []
    ridge_warnings = []
    drift_dominated = []
    basin_scores = []
    elite_tuning_count = 0
    
    for asset, raw_data in cache.items():
        data = raw_data.get('global', raw_data) if 'global' in raw_data else raw_data
        
        # Check for elite diagnostics in the result
        diags = data.get('diagnostics', {})
        if isinstance(diags, dict):
            elite = diags.get('elite_diagnostics', {})
            if elite:
                elite_tuning_count += 1
                
                if 'fragility_index' in elite:
                    fragility_indices.append(elite['fragility_index'])
                    if elite.get('fragility_warning', False):
                        fragility_warnings.append(asset)
                
                if 'condition_number' in elite:
                    cn = elite['condition_number']
                    if cn is not None and np.isfinite(cn):
                        condition_numbers.append(cn)
                
                # v2.0: Ridge detection
                if elite.get('is_ridge_optimum', False):
                    ridge_warnings.append(asset)
                
                if 'basin_score' in elite:
                    bs = elite['basin_score']
                    if np.isfinite(bs):
                        basin_scores.append(bs)
                
                # v2.0: Drift analysis
                drift_comp = elite.get('drift_component', [])
                noise_comp = elite.get('noise_component', [])
                if drift_comp and noise_comp:
                    total_var = sum(drift_comp) + sum(noise_comp)
                    if total_var > 1e-12:
                        drift_ratio = sum(drift_comp) / total_var
                        if drift_ratio > 0.6:  # Drift-dominated
                            drift_dominated.append(asset)
    
    if elite_tuning_count == 0:
        return
    
    # Calculate statistics
    avg_fragility = np.mean(fragility_indices) if fragility_indices else 0.0
    median_condition = np.median(condition_numbers) if condition_numbers else 1.0
    avg_basin_score = np.mean(basin_scores) if basin_scores else 1.0
    pct_low_fragility = sum(1 for f in fragility_indices if f < 0.3) / len(fragility_indices) * 100 if fragility_indices else 0
    pct_high_fragility = len(fragility_warnings) / len(fragility_indices) * 100 if fragility_indices else 0
    pct_ridges = len(ridge_warnings) / elite_tuning_count * 100 if elite_tuning_count > 0 else 0
    pct_drift = len(drift_dominated) / elite_tuning_count * 100 if elite_tuning_count > 0 else 0
    
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    section = Text()
    section.append("  🎯  ", style="bold bright_cyan")
    section.append("ELITE TUNING DIAGNOSTICS", style="bold bright_white")
    section.append(f"  (v2.0 Plateau-Optimal Selection)", style="dim")
    console.print(section)
    console.print()
    
    # Fragility summary
    fragility_text = Text()
    fragility_text.append("    Fragility Index (avg): ", style="dim")
    frag_color = "bright_green" if avg_fragility < 0.3 else "yellow" if avg_fragility < 0.5 else "indian_red1"
    fragility_text.append(f"{avg_fragility:.3f}", style=f"bold {frag_color}")
    console.print(fragility_text)
    
    # Condition number summary
    cond_text = Text()
    cond_text.append("    Hessian Condition (median): ", style="dim")
    cond_color = "bright_green" if median_condition < 1e4 else "yellow" if median_condition < 1e6 else "indian_red1"
    cond_text.append(f"{median_condition:.1e}", style=f"bold {cond_color}")
    console.print(cond_text)
    
    # v2.0: Basin score (ridge detection)
    basin_text = Text()
    basin_text.append("    Basin Score (avg): ", style="dim")
    basin_color = "bright_green" if avg_basin_score > 0.5 else "yellow" if avg_basin_score > 0.3 else "indian_red1"
    basin_text.append(f"{avg_basin_score:.3f}", style=f"bold {basin_color}")
    if pct_ridges > 0:
        basin_text.append(f"  ({pct_ridges:.0f}% ridge optima)", style="dim italic")
    console.print(basin_text)
    
    # Stability breakdown
    stability_text = Text()
    stability_text.append("    Parameter Stability: ", style="dim")
    stability_text.append(f"{pct_low_fragility:.0f}%", style="bold bright_green")
    stability_text.append(" stable  ·  ", style="dim")
    stability_text.append(f"{100 - pct_low_fragility - pct_high_fragility:.0f}%", style="bold yellow")
    stability_text.append(" moderate  ·  ", style="dim")
    stability_text.append(f"{pct_high_fragility:.0f}%", style="bold indian_red1")
    stability_text.append(" fragile", style="dim")
    console.print(stability_text)
    
    # v2.0: Drift analysis
    if pct_drift > 0:
        drift_text = Text()
        drift_text.append("    Drift Analysis: ", style="dim")
        drift_text.append(f"{pct_drift:.0f}%", style="bold yellow")
        drift_text.append(" drift-dominated (⚠ persistent parameter drift)", style="dim italic")
        console.print(drift_text)
    
    # Fragility warnings
    if fragility_warnings:
        console.print()
        warn_text = Text()
        warn_text.append("    ⚠ ", style="yellow")
        warn_text.append(f"{len(fragility_warnings)} high fragility: ", style="dim")
        displayed = fragility_warnings[:5]
        warn_text.append(", ".join(displayed), style="yellow")
        if len(fragility_warnings) > 5:
            warn_text.append(f" + {len(fragility_warnings) - 5} more", style="dim")
        console.print(warn_text)
    
    # Ridge warnings (v2.0)
    if ridge_warnings:
        ridge_text = Text()
        ridge_text.append("    ⚠ ", style="indian_red1")
        ridge_text.append(f"{len(ridge_warnings)} ridge optima: ", style="dim")
        displayed = ridge_warnings[:5]
        ridge_text.append(", ".join(displayed), style="indian_red1")
        if len(ridge_warnings) > 5:
            ridge_text.append(f" + {len(ridge_warnings) - 5} more", style="dim")
        console.print(ridge_text)
    
    console.print()


def render_tuning_header(prior_mean: float, prior_lambda: float, lambda_regime: float, console: Console = None, momentum_enabled: bool = True) -> None:
    if console is None:
        console = create_tuning_console()
    console.clear()
    console.print()
    console.print()
    title = Text()
    title.append("◆", style="bold bright_cyan")
    title.append("  K A L M A N   T U N E R", style="bold bright_white")
    console.print(Align.center(title))
    subtitle = Text("Hierarchical Regime-Conditional Maximum Likelihood", style="dim")
    console.print(Align.center(subtitle))
    console.print()
    now = datetime.now()
    cores = multiprocessing.cpu_count()
    ctx = Text()
    ctx.append(f"{now.strftime('%H:%M')}", style="bold white")
    ctx.append("  ·  ", style="dim")
    ctx.append(f"{cores} cores", style="dim")
    ctx.append("  ·  ", style="dim")
    ctx.append(f"{now.strftime('%b %d, %Y')}", style="dim")
    console.print(Align.center(ctx))
    console.print()
    priors = Table.grid(padding=(0, 4))
    priors.add_column(justify="right")
    priors.add_column(justify="left")
    priors.add_column(justify="right")
    priors.add_column(justify="left")
    priors.add_column(justify="right")
    priors.add_column(justify="left")
    priors.add_row(
        "[dim]q prior[/dim]", f"[white]N({prior_mean:.1f}, {prior_lambda:.1f})[/white]",
        "[dim]φ prior[/dim]", "[white]N(0, τ)[/white]",
        "[dim]λ regime[/dim]", f"[white]{lambda_regime:.3f}[/white]",
    )
    console.print(Align.center(priors))
    console.print()
    chips1 = Text()
    chips1.append("○ ", style="green")
    chips1.append("Gaussian", style="green")
    chips1.append("   ○ ", style="cyan")
    chips1.append("φ-Gaussian", style="cyan")
    chips1.append("   ○ ", style="magenta")
    chips1.append("φ-Student-t", style="magenta")
    chips1.append(" ", style="dim")
    chips1.append("(ν ∈ {4,6,8,12,20})", style="dim")
    console.print(Align.center(chips1))
    chips2 = Text()
    chips2.append("○ ", style="bright_magenta")
    chips2.append("φ-Skew-t", style="bright_magenta")
    chips2.append("   ○ ", style="bright_cyan")
    chips2.append("φ-NIG", style="bright_cyan")
    chips2.append("   ○ ", style="bright_yellow")
    chips2.append("GMM", style="bright_yellow")
    chips2.append("   ○ ", style="bright_blue")
    chips2.append("Hansen-λ", style="bright_blue")
    console.print(Align.center(chips2))
    chips3 = Text()
    chips3.append("○ ", style="red")
    chips3.append("EVT/GPD", style="red")
    chips3.append("   ○ ", style="orange1")
    chips3.append("Contaminated-t", style="orange1")
    chips3.append("   ○ ", style="bright_red")
    chips3.append("RiskTemp", style="bright_red")
    chips3.append("   ○ ", style="bright_green")
    console.print(Align.center(chips3))
    # Momentum and Fisher-Rao augmentation status
    chips4 = Text()
    if momentum_enabled:
        chips4.append("○ ", style="bright_yellow")
        chips4.append("Momentum", style="bright_yellow")
        chips4.append(" ", style="dim")
        chips4.append("(BMA augmentation)", style="dim")
        chips4.append("   ○ ", style="bright_cyan")
        chips4.append("Fisher-Rao", style="bright_cyan")
    else:
        chips4.append("○ ", style="dim")
        chips4.append("Momentum", style="dim")
        chips4.append(" ", style="dim")
        chips4.append("(disabled)", style="dim")
    console.print(Align.center(chips4))
    console.print(Align.center(Text(" " * 50)))
    console.print()


def render_tuning_progress_start(n_assets: int, n_workers: int, n_cached: int, cache_size: int, cache_path: str, console: Console = None) -> None:
    if console is None:
        console = create_tuning_console()
    console.print()
    console.print(Rule(style="dim", characters="─"))
    console.print()
    title = Text()
    title.append("▸ ", style="bright_yellow")
    title.append("ESTIMATION", style="bold white")
    console.print(title)
    console.print()
    stats = Text()
    stats.append("    ")
    stats.append(f"{n_assets}", style="bold bright_yellow")
    stats.append(" to process", style="dim")
    stats.append("   ·   ", style="dim")
    stats.append(f"{n_cached}", style="bold cyan")
    stats.append(" cached", style="dim")
    stats.append("   ·   ", style="dim")
    stats.append(f"{n_workers}", style="bold white")
    stats.append(" cores", style="dim")
    stats.append("   ·   ", style="dim")
    stats.append(f"{cache_size:,}", style="white")
    stats.append(" in cache", style="dim")
    console.print(stats)
    console.print()


def render_cache_status(cache_size: int, cache_path: str, console: Console = None) -> None:
    if console is None:
        console = create_tuning_console()
    filename = cache_path.split('/')[-1]
    console.print(f"  [dim]Cache:[/dim] [white]{cache_size:,}[/white] [dim]entries in[/dim] [white]{filename}[/white]")


def render_cache_update(cache_path: str, console: Console = None) -> None:
    if console is None:
        console = create_tuning_console()
    console.print(f"  [green]✓[/green] [dim]Saved[/dim]")


def render_asset_progress(asset: str, index: int, total: int, status: str, details: Optional[str] = None, console: Console = None) -> None:
    if console is None:
        console = create_tuning_console()
    icons = {'success': '[green]✓[/green]', 'cached': '[blue]○[/blue]', 'failed': '[red]✗[/red]', 'warning': '[yellow]![/yellow]'}
    icon = icons.get(status, '·')
    detail_str = f" [dim]{details}[/dim]" if details else ""
    console.print(f"    {icon} [white]{asset}[/white]{detail_str}")


def _get_status(fit_count: int, shrunk_count: int) -> str:
    if fit_count == 0:
        return "—"
    elif shrunk_count > 0:
        pct = shrunk_count / fit_count * 100 if fit_count > 0 else 0
        return f"{pct:.0f}%"
    return "✓"


def render_pdde_escalation_summary(escalation_summary: Dict[str, any], console: Console = None) -> None:
    """Render PIT-Driven Distribution Escalation summary with hierarchical level breakdown."""
    if console is None:
        console = create_tuning_console()
    total = escalation_summary.get('total', 0)
    if total == 0:
        return
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    section = Text()
    section.append("  🎯  ", style="bold bright_yellow")
    section.append("PIT CALIBRATION STATUS", style="bold bright_white")
    console.print(section)
    console.print()
    
    # Calibration status row
    calibrated = escalation_summary.get('calibrated', 0)
    calibrated_pct = escalation_summary.get('calibrated_pct', 0)
    warnings = escalation_summary.get('warnings', 0)
    critical = escalation_summary.get('critical', 0)
    
    status_row = Text()
    status_row.append("    Calibration: ", style="dim")
    status_row.append(f"{calibrated}", style="bold bright_green")
    status_row.append(f" ({calibrated_pct:.1f}%) passed", style="dim")
    status_row.append("  ·  ", style="dim")
    if warnings > 0:
        status_row.append(f"{warnings} warnings", style="yellow")
    if critical > 0:
        status_row.append("  ·  ", style="dim")
        status_row.append(f"{critical} critical", style="indian_red1")
    console.print(status_row)
    console.print()
    
    # Model distribution by BIC selection
    level_counts = escalation_summary.get('level_counts', {})
    
    if level_counts:
        console.print("    [dim]Model Selection (BIC-based, heavier tails when needed):[/dim]")
        console.print()
        
        bar_width = 25
        
        # Get escalation attempt stats
        nu_attempts = escalation_summary.get('nu_refinement_attempts', 0)
        nu_successes = escalation_summary.get('nu_refinement_successes', 0)
        nu_rate = escalation_summary.get('nu_refinement_success_rate', 0)
        
        evt_attempts = escalation_summary.get('evt_attempts', 0)
        evt_successes = escalation_summary.get('evt_successes', 0)
        evt_rate = escalation_summary.get('evt_success_rate', 0)
        
        gh_attempts = escalation_summary.get('gh_attempts', 0)
        gh_successes = escalation_summary.get('gh_successes', 0)
        gh_rate = escalation_summary.get('gh_success_rate', 0)
        
        tvvm_attempts = escalation_summary.get('tvvm_attempts', 0)
        tvvm_successes = escalation_summary.get('tvvm_successes', 0)
        tvvm_rate = escalation_summary.get('tvvm_success_rate', 0)
        
        mix_attempts = escalation_summary.get('mixture_attempts', 0)
        mix_successes = escalation_summary.get('mixture_successes', 0)
        mix_rate = escalation_summary.get('mixture_success_rate', 0)
        
        # Define levels with their display properties
        # (level_name, level_code, color, symbol, is_disabled, count_override, attempts, successes, rate, rate_label)
        # Note: Momentum models are now the primary models (February 2026)
        
        # Get momentum counts from level_counts
        gaussian_mom_count = level_counts.get('Gaussian+Momentum', 0)
        phi_gaussian_mom_count = level_counts.get('φ-Gaussian+Momentum', 0)
        phi_student_t_mom_count = level_counts.get('φ-Student-t+Momentum', 0)
        total_momentum = gaussian_mom_count + phi_gaussian_mom_count + phi_student_t_mom_count
        
        # Get base model counts (non-momentum) - these are now disabled
        gaussian_base_count = level_counts.get('Gaussian', 0)
        phi_gaussian_base_count = level_counts.get('φ-Gaussian', 0)
        phi_student_t_base_count = level_counts.get('φ-Student-t', 0)
        
        levels = [
            # Momentum models (primary - enabled)
            ('Gaussian+Momentum', 'M0', 'bright_green', '○', False, gaussian_mom_count, 0, 0, 0, None),
            ('φ-Gaussian+Momentum', 'M1', 'bright_cyan', '◇', False, phi_gaussian_mom_count, 0, 0, 0, None),
            ('φ-Student-t+Momentum', 'M2', 'bright_magenta', '●', False, phi_student_t_mom_count, 0, 0, 0, None),
            # Base Student-t (still enabled for non-momentum selection)
            ('φ-Student-t', 'L1', 'magenta', '●', False, phi_student_t_base_count, 0, 0, 0, None),
            ('φ-Student-t (ν-refined)', 'L2', 'bright_magenta', '◆', False, None, nu_attempts, nu_successes, nu_rate, 'improved'),
            ('EVT Tail Splice', 'L3', 'bright_red', '▲', False, None, evt_attempts, evt_successes, evt_rate, 'heavy'),
            ('Generalized Hyperbolic', 'L4', 'bright_cyan', '★', False, gh_successes, gh_attempts, gh_successes, gh_rate, 'improved'),
            ('TVVM', 'L5', 'yellow', '⚡', False, tvvm_successes, tvvm_attempts, tvvm_successes, tvvm_rate, 'improved'),
            # Disabled models at the bottom
            ('Gaussian', 'LD', 'dim', '○', True, gaussian_base_count, 0, 0, 0, None),
            ('φ-Gaussian', 'LD', 'dim', '◇', True, phi_gaussian_base_count, 0, 0, 0, None),
            ('K=2 Scale Mixture', 'LD', 'dim', '◈', True, None, mix_attempts, mix_successes, mix_rate, 'improved'),
        ]
        
        for level_name, level_code, color, symbol, is_disabled, count_override, attempts, successes, rate, rate_label in levels:
            # Use count_override if provided, otherwise get from level_counts
            if count_override is not None:
                count = count_override
            else:
                count = level_counts.get(level_name, 0)
            pct = count / total * 100 if total > 0 else 0
            filled = int(pct / 100 * bar_width)
            
            row = Text()
            row.append(f"      {symbol} ", style="dim" if is_disabled else color)
            row.append(f"{level_code} ", style="dim")
            
            # Display name - show "adaptive ν" instead of "ν-refined"
            display_name = level_name
            if level_name == 'φ-Student-t (ν-refined)':
                display_name = 'φ-Student-t (adaptive ν)'
            elif level_name == 'φ-Student-t+Momentum (ν-refined)':
                display_name = 'φ-Student-t+Momentum (adaptive ν)'
            
            if is_disabled:
                row.append(f"{display_name:<28}", style="dim")
                row.append("░" * bar_width, style="dim")
                row.append(f"  {count:>4}  ({pct:>5.1f}%)", style="dim")
                # Add PIT attempt stats for disabled levels too
                if attempts > 0:
                    row.append(f"  [{successes}/{attempts} {rate:.0f}%]", style="dim italic")
                row.append("  [disabled]", style="dim italic")
            else:
                row.append(f"{display_name:<28}", style=color if count > 0 else "dim")
                row.append("█" * filled, style=color)
                row.append("░" * (bar_width - filled), style="dim")
                row.append(f"  {count:>4}  ({pct:>5.1f}%)", style="white" if count > 0 else "dim")
                
                # Add PIT improvement stats if attempts were made
                if attempts > 0 and rate_label:
                    rate_str = f"  [{successes}/{attempts} {rate:.0f}% {rate_label}]"
                    row.append(rate_str, style="dim italic")
                elif count > 0:
                    # Add appropriate annotation based on model type
                    if 'Momentum' in level_name:
                        row.append("  ↑ momentum augmented", style="dim italic")
                    elif 'Student-t' in level_name and 'base' not in level_name.lower():
                        row.append("  ↑ heavier tails", style="dim italic")
            
            console.print(row)
        
        console.print()
        
        # Show momentum model summary (February 2026)
        if total_momentum > 0:
            momentum_pct = total_momentum / total * 100 if total > 0 else 0
            mom_summary = Text()
            mom_summary.append("    Momentum models: ", style="dim")
            mom_summary.append(f"{total_momentum}", style="bold bright_yellow")
            mom_summary.append(f" ({momentum_pct:.1f}% of assets)", style="dim")
            console.print(mom_summary)
        
        # Show how many assets needed heavier tails
        escalations = escalation_summary.get('escalations_triggered', 0)
        escalation_rate = escalation_summary.get('escalation_rate', 0)
        if escalations > 0:
            esc_row = Text()
            esc_row.append("    Heavier tails selected: ", style="dim")
            esc_row.append(f"{escalations}", style="bold bright_cyan")
            esc_row.append(f" ({escalation_rate:.1f}% of assets)", style="dim")
            console.print(esc_row)
        
        # Show calibration issue summary
        if critical > 0:
            console.print()
            issue_row = Text()
            issue_row.append("    ⚠ ", style="indian_red1")
            issue_row.append(f"{critical} assets with PIT p < 0.01", style="indian_red1")
            issue_row.append(" — consider enabling additional escalation models", style="dim")
            console.print(issue_row)
        
        console.print()


def render_tuning_summary(
    total_assets: int, new_estimates: int, reused_cached: int, failed: int,
    calibration_warnings: int, gaussian_count: int, student_t_count: int,
    regime_tuning_count: int, lambda_regime: float, regime_fit_counts: Dict[int, int],
    regime_shrunk_counts: Dict[int, int], collapse_warnings: int, cache_path: str,
    regime_model_breakdown: Optional[Dict[int, Dict[str, int]]] = None,
    mixture_attempted_count: int = 0, mixture_selected_count: int = 0,
    nu_refinement_attempted_count: int = 0, nu_refinement_improved_count: int = 0,
    gh_attempted_count: int = 0, gh_selected_count: int = 0,
    tvvm_attempted_count: int = 0, tvvm_selected_count: int = 0,
    phi_gaussian_count: int = 0, phi_student_t_count: int = 0,
    phi_skew_t_count: int = 0, phi_nig_count: int = 0,
    gmm_fitted_count: int = 0, hansen_fitted_count: int = 0,
    hansen_left_skew_count: int = 0, hansen_right_skew_count: int = 0,
    evt_fitted_count: int = 0, evt_heavy_tail_count: int = 0,
    evt_moderate_tail_count: int = 0, evt_light_tail_count: int = 0,
    contaminated_t_count: int = 0, recalibration_applied_count: int = 0,
    calibrated_trust_count: int = 0, avg_effective_trust: float = 0.0,
    low_trust_count: int = 0, high_trust_count: int = 0,
    # Momentum augmentation counts (specific breakdown)
    momentum_gaussian_count: int = 0,  # Gaussian with momentum (no phi)
    momentum_phi_gaussian_count: int = 0,  # φ-Gaussian with momentum
    momentum_phi_student_t_count: int = 0,  # φ-Student-t with momentum
    momentum_total_count: int = 0,
    # Enhanced Student-t counts (February 2026)
    vov_enhanced_count: int = 0,  # Vol-of-Vol enhanced
    two_piece_count: int = 0,  # Two-Piece asymmetric tails
    mixture_t_count: int = 0,  # Two-Component mixture
    # Volatility estimator counts (February 2026)
    gk_vol_count: int = 0,  # Garman-Klass volatility
    har_vol_count: int = 0,  # HAR volatility
    ewma_vol_count: int = 0,  # EWMA volatility (fallback)
    # CRPS model selection (February 2026)
    crps_computed_count: int = 0,  # Models with CRPS computed
    crps_regime_aware_count: int = 0,  # Assets using regime-aware CRPS weights
    # Legacy parameter for backward compatibility
    momentum_student_t_count: int = 0,
    console: Console = None
) -> None:
    """Render tuning summary with model selection breakdown."""
    if console is None:
        console = create_tuning_console()
    
    console.print()
    console.print()
    
    header_text = Text(justify="center")
    header_text.append("\n", style="")
    header_text.append("✓ ", style="bold bright_green")
    header_text.append("TUNING COMPLETE", style="bold bright_white")
    header_text.append("\n", style="")
    header_panel = Panel(Align.center(header_text), box=box.ROUNDED, border_style="bright_green", padding=(0, 4), width=40)
    console.print(Align.center(header_panel))
    console.print()
    
    # Metrics row
    metrics_table = Table(show_header=False, box=None, padding=(0, 4), expand=False)
    metrics_table.add_column(justify="center")
    metrics_table.add_column(justify="center")
    metrics_table.add_column(justify="center")
    metrics_table.add_column(justify="center")
    
    def metric_text(value: int, label: str, color: str = "white") -> Text:
        t = Text(justify="center")
        t.append(f"{value:,}\n", style=f"bold {color}")
        t.append(label, style="dim")
        return t
    
    failed_color = "indian_red1" if failed > 0 else "dim"
    metrics_table.add_row(
        metric_text(total_assets, "Total", "bright_white"),
        metric_text(new_estimates, "New", "bright_green"),
        metric_text(reused_cached, "Cached", "bright_cyan"),
        metric_text(failed, "Failed", failed_color),
    )
    console.print(Align.center(metrics_table))
    console.print()
    console.print()
    
    # Model selection section
    total_models = gaussian_count + student_t_count
    if total_models > 0:
        console.print(Rule(style="dim"))
        console.print()
        section = Text()
        section.append("  📈  ", style="bold bright_cyan")
        section.append("MODEL SELECTION", style="bold bright_white")
        console.print(section)
        console.print()
        
        # Calculate non-momentum counts (base models only)
        non_mom_gaussian = gaussian_count - momentum_gaussian_count - momentum_phi_gaussian_count
        non_mom_phi_gaussian = phi_gaussian_count - momentum_phi_gaussian_count
        non_mom_phi_student_t = phi_student_t_count - momentum_phi_student_t_count
        
        # Ensure non-negative
        non_mom_gaussian = max(0, non_mom_gaussian)
        non_mom_phi_gaussian = max(0, non_mom_phi_gaussian)
        non_mom_phi_student_t = max(0, non_mom_phi_student_t)
        
        bar_width = 30
        
        # ═══════════════════════════════════════════════════════════════════
        # BASE MODELS (Non-Momentum) - Gaussian/φ-Gaussian DISABLED
        # ═══════════════════════════════════════════════════════════════════
        base_section = Text()
        base_section.append("    ▸ Base Models (Non-Momentum)", style="bold dim")
        console.print(base_section)
        console.print()
        
        # Gaussian (non-momentum) - DISABLED
        gauss_pct = non_mom_gaussian / total_models * 100 if total_models > 0 else 0
        gauss_row = Text()
        gauss_row.append("      ○ ", style="dim")
        gauss_row.append(f"{'Gaussian':<18} ", style="dim")
        gauss_row.append("░" * bar_width, style="dim")
        gauss_row.append(f"  {non_mom_gaussian:>4}", style="dim")
        gauss_row.append(f"  ({gauss_pct:>4.1f}%)", style="dim")
        gauss_row.append("  [disabled]", style="dim italic")
        console.print(gauss_row)
        
        # φ-Gaussian (non-momentum) - DISABLED
        phi_g_pct = non_mom_phi_gaussian / total_models * 100 if total_models > 0 else 0
        phi_g_row = Text()
        phi_g_row.append("      ◇ ", style="dim")
        phi_g_row.append(f"{'φ-Gaussian':<18} ", style="dim")
        phi_g_row.append("░" * bar_width, style="dim")
        phi_g_row.append(f"  {non_mom_phi_gaussian:>4}", style="dim")
        phi_g_row.append(f"  ({phi_g_pct:>4.1f}%)", style="dim")
        phi_g_row.append("  [disabled]", style="dim italic")
        console.print(phi_g_row)
        
        # φ-Student-t (non-momentum) - ENABLED
        phi_st_pct = non_mom_phi_student_t / total_models * 100 if total_models > 0 else 0
        phi_st_filled = int(phi_st_pct / 100 * bar_width)
        phi_st_style = "magenta" if non_mom_phi_student_t > 0 else "dim"
        phi_st_row = Text()
        phi_st_row.append("      ● ", style=phi_st_style)
        phi_st_row.append(f"{'φ-Student-t':<18} ", style=phi_st_style)
        phi_st_row.append("█" * phi_st_filled, style=phi_st_style)
        phi_st_row.append("░" * (bar_width - phi_st_filled), style="dim")
        phi_st_row.append(f"  {non_mom_phi_student_t:>4}", style="bold white" if non_mom_phi_student_t > 0 else "dim")
        phi_st_row.append(f"  ({phi_st_pct:>4.1f}%)", style="dim")
        console.print(phi_st_row)
        
        # ═══════════════════════════════════════════════════════════════════
        # MOMENTUM MODELS
        # ═══════════════════════════════════════════════════════════════════
        console.print()
        mom_section = Text()
        mom_section.append("    ▸ Momentum Models", style="bold dim")
        console.print(mom_section)
        console.print()
        
        # Gaussian+Momentum (no phi)
        mom_g_pct = momentum_gaussian_count / total_models * 100 if total_models > 0 else 0
        mom_g_filled = int(mom_g_pct / 100 * bar_width)
        mom_g_style = "bright_green" if momentum_gaussian_count > 0 else "dim"
        mom_g_row = Text()
        mom_g_row.append("      ○ ", style=mom_g_style)
        mom_g_row.append(f"{'Gaussian+Momentum':<22} ", style=mom_g_style)
        mom_g_row.append("█" * mom_g_filled, style=mom_g_style)
        mom_g_row.append("░" * (bar_width - mom_g_filled), style="dim")
        mom_g_row.append(f"  {momentum_gaussian_count:>4}", style="bold white" if momentum_gaussian_count > 0 else "dim")
        mom_g_row.append(f"  ({mom_g_pct:>4.1f}%)", style="dim")
        console.print(mom_g_row)
        
        # φ-Gaussian+Momentum
        mom_phi_g_pct = momentum_phi_gaussian_count / total_models * 100 if total_models > 0 else 0
        mom_phi_g_filled = int(mom_phi_g_pct / 100 * bar_width)
        mom_phi_g_style = "bright_cyan" if momentum_phi_gaussian_count > 0 else "dim"
        mom_phi_g_row = Text()
        mom_phi_g_row.append("      ◇ ", style=mom_phi_g_style)
        mom_phi_g_row.append(f"{'φ-Gaussian+Momentum':<22} ", style=mom_phi_g_style)
        mom_phi_g_row.append("█" * mom_phi_g_filled, style=mom_phi_g_style)
        mom_phi_g_row.append("░" * (bar_width - mom_phi_g_filled), style="dim")
        mom_phi_g_row.append(f"  {momentum_phi_gaussian_count:>4}", style="bold white" if momentum_phi_gaussian_count > 0 else "dim")
        mom_phi_g_row.append(f"  ({mom_phi_g_pct:>4.1f}%)", style="dim")
        console.print(mom_phi_g_row)
        
        # φ-Student-t+Momentum
        mom_phi_st_pct = momentum_phi_student_t_count / total_models * 100 if total_models > 0 else 0
        mom_phi_st_filled = int(mom_phi_st_pct / 100 * bar_width)
        mom_phi_st_style = "bright_magenta" if momentum_phi_student_t_count > 0 else "dim"
        mom_phi_st_row = Text()
        mom_phi_st_row.append("      ● ", style=mom_phi_st_style)
        mom_phi_st_row.append(f"{'φ-Student-t+Momentum':<22} ", style=mom_phi_st_style)
        mom_phi_st_row.append("█" * mom_phi_st_filled, style=mom_phi_st_style)
        mom_phi_st_row.append("░" * (bar_width - mom_phi_st_filled), style="dim")
        mom_phi_st_row.append(f"  {momentum_phi_student_t_count:>4}", style="bold white" if momentum_phi_student_t_count > 0 else "dim")
        mom_phi_st_row.append(f"  ({mom_phi_st_pct:>4.1f}%)", style="dim")
        console.print(mom_phi_st_row)
        
        # Momentum summary
        if momentum_total_count > 0:
            console.print()
            mom_sum_row = Text()
            mom_sum_row.append("      ─────────────────────────────────────────────────────────", style="dim")
            console.print(mom_sum_row)
            mom_total_row = Text()
            mom_total_row.append("      Total Momentum: ", style="dim")
            mom_total_row.append(f"{momentum_total_count}", style="bold bright_yellow")
            mom_total_pct = momentum_total_count / total_models * 100 if total_models > 0 else 0
            mom_total_row.append(f" ({mom_total_pct:.1f}%)", style="dim")
            console.print(mom_total_row)
        
        # ═══════════════════════════════════════════════════════════════════
        # ENHANCED STUDENT-T MODELS (Vol-of-Vol, Two-Piece, Mixture)
        # ═══════════════════════════════════════════════════════════════════
        enhanced_total = vov_enhanced_count + two_piece_count + mixture_t_count
        if enhanced_total > 0:
            console.print()
            enhanced_section = Text()
            enhanced_section.append("    ▸ Enhanced Student-t", style="bold dim")
            console.print(enhanced_section)
            console.print()
            
            # Vol-of-Vol enhanced
            vov_pct = vov_enhanced_count / total_models * 100 if total_models > 0 else 0
            vov_filled = int(vov_pct / 100 * bar_width)
            vov_style = "bright_magenta" if vov_enhanced_count > 0 else "dim"
            vov_row = Text()
            vov_row.append("      ◎ ", style=vov_style)
            vov_row.append(f"{'Vol-of-Vol+Mom':<22} ", style=vov_style)
            vov_row.append("█" * vov_filled, style=vov_style)
            vov_row.append("░" * (bar_width - vov_filled), style="dim")
            vov_row.append(f"  {vov_enhanced_count:>4}", style="bold white" if vov_enhanced_count > 0 else "dim")
            vov_row.append(f"  ({vov_pct:>4.1f}%)", style="dim")
            console.print(vov_row)
            
            # Two-Piece asymmetric
            tp_pct = two_piece_count / total_models * 100 if total_models > 0 else 0
            tp_filled = int(tp_pct / 100 * bar_width)
            tp_style = "yellow" if two_piece_count > 0 else "dim"
            tp_row = Text()
            tp_row.append("      ◐ ", style=tp_style)
            tp_row.append(f"{'Two-Piece-t+Mom':<22} ", style=tp_style)
            tp_row.append("█" * tp_filled, style=tp_style)
            tp_row.append("░" * (bar_width - tp_filled), style="dim")
            tp_row.append(f"  {two_piece_count:>4}", style="bold white" if two_piece_count > 0 else "dim")
            tp_row.append(f"  ({tp_pct:>4.1f}%)", style="dim")
            console.print(tp_row)
            
            # Two-Component mixture
            mix_pct = mixture_t_count / total_models * 100 if total_models > 0 else 0
            mix_filled = int(mix_pct / 100 * bar_width)
            mix_style = "bright_green" if mixture_t_count > 0 else "dim"
            mix_row = Text()
            mix_row.append("      ◉ ", style=mix_style)
            mix_row.append(f"{'Mixture-t+Mom':<22} ", style=mix_style)
            mix_row.append("█" * mix_filled, style=mix_style)
            mix_row.append("░" * (bar_width - mix_filled), style="dim")
            mix_row.append(f"  {mixture_t_count:>4}", style="bold white" if mixture_t_count > 0 else "dim")
            mix_row.append(f"  ({mix_pct:>4.1f}%)", style="dim")
            console.print(mix_row)
            
            # Enhanced summary
            console.print()
            enh_sum_row = Text()
            enh_sum_row.append("      ─────────────────────────────────────────────────────────", style="dim")
            console.print(enh_sum_row)
            enh_total_row = Text()
            enh_total_row.append("      Total Enhanced: ", style="dim")
            enh_total_row.append(f"{enhanced_total}", style="bold bright_cyan")
            enh_total_pct = enhanced_total / total_models * 100 if total_models > 0 else 0
            enh_total_row.append(f" ({enh_total_pct:.1f}%)", style="dim")
            console.print(enh_total_row)
        
        # ═══════════════════════════════════════════════════════════════════
        # OTHER VARIANTS (φ-Skew-t, φ-NIG) - only show if any exist
        # ═══════════════════════════════════════════════════════════════════
        if phi_skew_t_count > 0 or phi_nig_count > 0:
            console.print()
            other_section = Text()
            other_section.append("    ▸ Other Variants", style="bold dim")
            console.print(other_section)
            console.print()
            
            # φ-Skew-t
            skt_pct = phi_skew_t_count / total_models * 100 if total_models > 0 else 0
            skt_filled = int(skt_pct / 100 * bar_width)
            skt_style = "bright_cyan" if phi_skew_t_count > 0 else "dim"
            skt_row = Text()
            skt_row.append("      ◆ ", style=skt_style)
            skt_row.append(f"{'φ-Skew-t':<18} ", style=skt_style)
            skt_row.append("█" * skt_filled, style=skt_style)
            skt_row.append("░" * (bar_width - skt_filled), style="dim")
            skt_row.append(f"  {phi_skew_t_count:>4}", style="bold white" if phi_skew_t_count > 0 else "dim")
            skt_row.append(f"  ({skt_pct:>4.1f}%)", style="dim")
            console.print(skt_row)
            
            # φ-NIG
            nig_pct = phi_nig_count / total_models * 100 if total_models > 0 else 0
            nig_filled = int(nig_pct / 100 * bar_width)
            nig_style = "bright_yellow" if phi_nig_count > 0 else "dim"
            nig_row = Text()
            nig_row.append("      ★ ", style=nig_style)
            nig_row.append(f"{'φ-NIG':<18} ", style=nig_style)
            nig_row.append("█" * nig_filled, style=nig_style)
            nig_row.append("░" * (bar_width - nig_filled), style="dim")
            nig_row.append(f"  {phi_nig_count:>4}", style="bold white" if phi_nig_count > 0 else "dim")
            nig_row.append(f"  ({nig_pct:>4.1f}%)", style="dim")
            console.print(nig_row)
        
        # ═══════════════════════════════════════════════════════════════════
        # AUGMENTATION LAYERS
        # ═══════════════════════════════════════════════════════════════════
        console.print()
        aug_section = Text()
        aug_section.append("    ▸ Augmentation Layers", style="bold dim")
        console.print(aug_section)
        console.print()
        
        gmm_pct = gmm_fitted_count / total_models * 100 if total_models > 0 else 0
        gmm_filled = int(gmm_pct / 100 * bar_width)
        gmm_row = Text()
        gmm_row.append("      ◈ ", style="bright_blue")
        gmm_row.append(f"{'GMM (2-State)':<14} ", style="bright_blue")
        gmm_row.append("█" * gmm_filled, style="bright_blue")
        gmm_row.append("░" * (bar_width - gmm_filled), style="dim")
        gmm_row.append(f"  {gmm_fitted_count:>4}", style="bold white")
        gmm_row.append(f"  ({gmm_pct:>4.1f}%)", style="dim")
        if gmm_fitted_count == 0:
            gmm_row.append("  [disabled]", style="dim italic")
        console.print(gmm_row)
        
        hansen_pct = hansen_fitted_count / total_models * 100 if total_models > 0 else 0
        hansen_filled = int(hansen_pct / 100 * bar_width)
        hansen_row = Text()
        hansen_row.append("      λ ", style="bright_cyan")
        hansen_row.append(f"{'Hansen-λ':<14} ", style="bright_cyan")
        hansen_row.append("█" * hansen_filled, style="bright_cyan")
        hansen_row.append("░" * (bar_width - hansen_filled), style="dim")
        hansen_row.append(f"  {hansen_fitted_count:>4}", style="bold white")
        hansen_row.append(f"  ({hansen_pct:>4.1f}%)", style="dim")
        if hansen_fitted_count > 0 and (hansen_left_skew_count > 0 or hansen_right_skew_count > 0):
            hansen_row.append(f"  [←{hansen_left_skew_count}/→{hansen_right_skew_count}]", style="dim")
        elif hansen_fitted_count == 0:
            hansen_row.append("  [0 fitted]", style="dim italic")
        console.print(hansen_row)
        
        evt_pct = evt_fitted_count / total_models * 100 if total_models > 0 else 0
        evt_filled = int(evt_pct / 100 * bar_width)
        evt_row = Text()
        evt_row.append("      ξ ", style="indian_red1")
        evt_row.append(f"{'EVT/GPD':<14} ", style="indian_red1")
        evt_row.append("█" * evt_filled, style="indian_red1")
        evt_row.append("░" * (bar_width - evt_filled), style="dim")
        evt_row.append(f"  {evt_fitted_count:>4}", style="bold white")
        evt_row.append(f"  ({evt_pct:>4.1f}%)", style="dim")
        if evt_fitted_count > 0 and (evt_heavy_tail_count > 0 or evt_moderate_tail_count > 0 or evt_light_tail_count > 0):
            evt_row.append(f"  [H:{evt_heavy_tail_count}/M:{evt_moderate_tail_count}/L:{evt_light_tail_count}]", style="dim")
        elif evt_fitted_count == 0:
            evt_row.append("  [0 fitted]", style="dim italic")
        console.print(evt_row)
        
        cst_pct = contaminated_t_count / total_models * 100 if total_models > 0 else 0
        cst_filled = int(cst_pct / 100 * bar_width)
        cst_row = Text()
        cst_row.append("      ⚠ ", style="yellow")
        cst_row.append(f"{'Contaminated-t':<14} ", style="yellow")
        cst_row.append("█" * cst_filled, style="yellow")
        cst_row.append("░" * (bar_width - cst_filled), style="dim")
        cst_row.append(f"  {contaminated_t_count:>4}", style="bold white")
        cst_row.append(f"  ({cst_pct:>4.1f}%)", style="dim")
        if contaminated_t_count == 0:
            cst_row.append("  [0 fitted]", style="dim italic")
        console.print(cst_row)
        
        console.print()
        console.print()
    
    # Calibrated Trust Authority section
    console.print(Rule(style="dim"))
    console.print()
    section = Text()
    section.append("  🎯  ", style="bold bright_cyan")
    section.append("CALIBRATED TRUST AUTHORITY", style="bold bright_white")
    console.print(section)
    console.print()
    
    if calibrated_trust_count > 0 or recalibration_applied_count > 0:
        # Isotonic Recalibration subsection
        recal_section = Text()
        recal_section.append("    ◈ ", style="bright_cyan")
        recal_section.append("Isotonic Recalibration", style="bright_cyan")
        console.print(recal_section)
        
        recal_row = Text()
        recal_row.append("      Applied: ", style="dim")
        recal_row.append(f"{recalibration_applied_count if recalibration_applied_count > 0 else calibrated_trust_count}", style="bold bright_white")
        recal_row.append(" assets", style="dim")
        console.print(recal_row)
        console.print()
        
        # Trust Distribution subsection
        trust_section = Text()
        trust_section.append("    ◉ ", style="bright_magenta")
        trust_section.append("Trust Distribution", style="bright_magenta")
        console.print(trust_section)
        
        trust_row = Text()
        trust_row.append("      Computed: ", style="dim")
        trust_row.append(f"{calibrated_trust_count}", style="bold bright_white")
        trust_row.append("  ·  Avg: ", style="dim")
        trust_row.append(f"{avg_effective_trust:.1%}", style="bold bright_white")
        console.print(trust_row)
        
        if low_trust_count > 0 or high_trust_count > 0:
            trust_dist = Text()
            trust_dist.append("      High (≥70%): ", style="dim")
            trust_dist.append(f"{high_trust_count}", style="bright_green")
            trust_dist.append("  ·  Low (<30%): ", style="dim")
            trust_dist.append(f"{low_trust_count}", style="indian_red1")
            console.print(trust_dist)
    else:
        # Show hint when no trust data available
        hint_row = Text()
        hint_row.append("    ⚡ Trust computed at signal time (based on PIT calibration)", style="dim")
        console.print(hint_row)
    console.print()
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # REGIME COVERAGE - Show regime distribution with model breakdown
    # ═══════════════════════════════════════════════════════════════════════════════
    console.print(Rule(style="dim"))
    console.print()
    
    section = Text()
    section.append("  🎯  ", style="bold bright_cyan")
    section.append("REGIME COVERAGE", style="bold bright_white")
    console.print(section)
    console.print()
    
    # Regime names and colors
    regime_names = ["LOW_VOL_TREND", "HIGH_VOL_TREND", "LOW_VOL_RANGE", "HIGH_VOL_RANGE", "CRISIS_JUMP"]
    regime_short = ["LV Trend", "HV Trend", "LV Range", "HV Range", "Crisis"]
    regime_colors_list = ["bright_cyan", "yellow", "bright_green", "orange1", "indian_red1"]
    regime_icons = ["◇", "◆", "○", "●", "⚠"]
    
    max_fits = max(regime_fit_counts.values()) if regime_fit_counts.values() else 1
    
    # Define STANDARD model columns that ALWAYS appear
    # Pure Gaussian and φ-Gaussian are DISABLED (February 2026) - only momentum versions shown
    # Format: (model_key, header, color, min_width)
    STANDARD_MODEL_COLUMNS = [
        # Momentum models (Gaussian-based) - momentum versions replace pure base models
        ("Gaussian+Mom", "G+M", "bright_green", 5),
        ("φ-Gaussian+Mom", "φG+M", "bright_cyan", 5),
        # Student-t by ν (non-momentum - still enabled as they have value)
        ("φ-t(ν=4)", "t4", "magenta", 4),
        ("φ-t(ν=6)", "t6", "magenta", 4),
        ("φ-t(ν=8)", "t8", "magenta", 4),
        ("φ-t(ν=12)", "t12", "magenta", 4),
        ("φ-t(ν=20)", "t20", "magenta", 4),
        # Momentum Student-t
        ("φ-Student-t+Mom", "t+M", "bright_magenta", 5),
        # Unified Student-t (February 2026 - Elite Architecture)
        ("φ-t-Unified-4", "U4", "bright_yellow", 3),
        ("φ-t-Unified-8", "U8", "bright_yellow", 3),
        ("φ-t-Unified-20", "U20", "bright_yellow", 4),
        # Augmentation layers
        ("Hansen-λ", "Hλ", "cyan", 5),
        ("EVT", "EVT", "indian_red1", 5),
        ("CST", "CST", "yellow", 5),
    ]
    
    # Helper to normalize model keys for comparison
    def normalize_model_key(m):
        # Check momentum models first (before base model checks)
        # φ-Student-t+Mom (momentum Student-t)
        if "Student-t+Mom" in m or "φ-Student-t+Mom" in m:
            return "φ-Student-t+Mom"
        if "+Mom" in m and ("Student" in m or "t(" in m or "phi_student" in m.lower()):
            return "φ-Student-t+Mom"
        # φ-Gaussian+Mom (momentum phi-gaussian)
        if "φ-Gaussian+Mom" in m or "kalman_phi_gaussian_momentum" in m.lower():
            return "φ-Gaussian+Mom"
        # Gaussian+Mom (momentum gaussian, no phi)
        if "Gaussian+Mom" in m or ("kalman_gaussian_momentum" in m.lower() and "phi" not in m.lower()):
            return "Gaussian+Mom"
        if "+Mom" in m and "Student" not in m and "t(" not in m:
            if "φ" in m or "phi" in m.lower():
                return "φ-Gaussian+Mom"
            return "Gaussian+Mom"
        # Unified Student-t models (February 2026 - Elite Architecture)
        if "unified" in m.lower():
            if "nu_4" in m or "_4" in m:
                return "φ-t-Unified-4"
            if "nu_20" in m or "_20" in m:
                return "φ-t-Unified-20"
            return "φ-t-Unified-8"
        # Student-t variants by ν
        if m.startswith("φ-t(ν="):
            return m
        if m.startswith("φ-Skew-t"):
            return "φ-Skew-t"
        if m.startswith("φ-NIG"):
            return "φ-NIG"
        # Augmentation layers
        if "GMM" in m:
            return "GMM"
        if "Hλ" in m or "Hansen" in m:
            return "Hansen-λ"
        if "EVT" in m:
            return "EVT"
        if "CST" in m:
            return "CST"
        return m
    
    # Helper to find count for a column
    def get_model_count(r_breakdown, col_key):
        if col_key in r_breakdown:
            return r_breakdown[col_key]
        norm_col = normalize_model_key(col_key)
        total = 0
        for actual_key, count in r_breakdown.items():
            if normalize_model_key(actual_key) == norm_col:
                total += count
        return total
    
    # Create elegant table
    table = Table(
        show_header=True,
        header_style="bold white",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1),
        row_styles=["", "on grey7"],
    )
    table.add_column("Regime", width=12)
    table.add_column("Fits", justify="right", width=5)
    table.add_column("Distribution", width=18)
    
    # Track which columns we add for row building
    column_model_keys = []
    
    # Add STANDARD columns
    for model_key, header, color, width in STANDARD_MODEL_COLUMNS:
        table.add_column(header, justify="right", width=width, style=color)
        column_model_keys.append(model_key)
    
    for i, (name, short, color, icon) in enumerate(zip(regime_names, regime_short, regime_colors_list, regime_icons)):
        fit_count = regime_fit_counts.get(i, 0)
        
        # Create visual bar
        if fit_count == 0:
            bar = "[dim]" + "─" * 18 + "[/]"
        else:
            filled = int(fit_count / max_fits * 18) if max_fits > 0 else 0
            bar = f"[{color}]{'━' * filled}[/{color}][dim]{'─' * (18 - filled)}[/]"
        
        # Build row
        row = [
            f"[{color}]{icon} {short}[/{color}]",
            f"[bold]{fit_count}[/]" if fit_count > 0 else "[dim]0[/]",
            bar,
        ]
        
        # Add counts for each column
        r_breakdown = regime_model_breakdown.get(i, {}) if regime_model_breakdown else {}
        for col_key in column_model_keys:
            count = get_model_count(r_breakdown, col_key)
            if count > 0:
                row.append(f"{count}")
            else:
                row.append("[dim]—[/]")
        
        table.add_row(*row)
    
    console.print(table)
    console.print()
    
    # Warnings
    if collapse_warnings > 0 or calibration_warnings > 0:
        warnings_text = Text()
        warnings_text.append("    ", style="")
        if collapse_warnings > 0:
            warnings_text.append("⚠ ", style="yellow")
            warnings_text.append(f"{collapse_warnings} collapse", style="dim")
            if calibration_warnings > 0:
                warnings_text.append("   ·   ", style="dim")
        if calibration_warnings > 0:
            warnings_text.append("⚠ ", style="yellow")
            warnings_text.append(f"{calibration_warnings} calibration", style="dim")
        console.print(warnings_text)
        console.print()
    
    console.print()


def render_parameter_table(cache: Dict[str, Dict], console: Console = None) -> None:
    """Render parameter table for tuned assets showing all parameters including unified model columns."""
    if console is None:
        console = create_tuning_console()
    if not cache:
        return
    
    def _model_label(data: dict) -> str:
        """Get full model name for display."""
        if 'global' in data:
            data = data['global']
        phi_val = data.get('phi')
        noise_model = data.get('noise_model', 'gaussian')
        best_model = data.get('best_model_by_bic', noise_model)
        nu_val = data.get('nu')
        
        # Check for unified model first
        if data.get('unified_model') or (best_model and 'unified' in str(best_model).lower()):
            nu_str = f"ν={int(nu_val)}" if nu_val else "ν=8"
            return f"φ-t-Unified({nu_str})"
        
        # Check for specific Student-t variants
        if noise_model and 'student_t' in noise_model.lower():
            # Extract ν from noise_model or use data value
            if 'nu_' in noise_model:
                try:
                    nu_from_name = int(noise_model.split('nu_')[-1].split('_')[0])
                    nu_str = f"ν={nu_from_name}"
                except:
                    nu_str = f"ν={int(nu_val)}" if nu_val else ""
            else:
                nu_str = f"ν={int(nu_val)}" if nu_val else ""
            
            # Check for momentum
            if '_momentum' in noise_model.lower() or data.get('momentum_augmented'):
                return f"φ-t({nu_str})+Mom" if nu_str else "φ-t+Mom"
            
            if phi_val is not None:
                return f"φ-t({nu_str})" if nu_str else "φ-t"
            return f"t({nu_str})" if nu_str else "t"
        
        # Gaussian variants
        if noise_model == 'kalman_phi_gaussian' or phi_val is not None:
            if '_momentum' in str(noise_model).lower() or data.get('momentum_augmented'):
                return 'φ-Gauss+Mom'
            return 'φ-Gaussian'
        
        if '_momentum' in str(noise_model).lower() or data.get('momentum_augmented'):
            return 'Gauss+Mom'
        return 'Gaussian'
    
    def _get_q_for_sort(data):
        if 'global' in data:
            return data['global'].get('q', 0)
        return data.get('q', 0)
    
    # Group by model type
    groups: Dict[str, List] = {}
    for asset, data in cache.items():
        model = _model_label(data)
        if model not in groups:
            groups[model] = []
        groups[model].append((asset, data))
    
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    section = Text()
    section.append("  📊  ", style="bold bright_cyan")
    section.append("TUNED PARAMETERS", style="bold bright_white")
    section.append(f"  ({len(cache)} assets)", style="dim")
    console.print(section)
    console.print()
    
    # Sort assets by model family, then by q descending
    sorted_assets = sorted(
        cache.items(),
        key=lambda x: (_model_label(x[1]), -_get_q_for_sort(x[1]))
    )
    
    # Create table with adjusted column widths for full model names
    table = Table(
        show_header=True,
        header_style="bold white",
        border_style="dim",
        box=box.SIMPLE,
        padding=(0, 1),
        collapse_padding=True,
    )
    
    table.add_column("Asset", style="bold white", width=12, no_wrap=True)
    table.add_column("Model", style="cyan", width=18, no_wrap=True)  # Wider for full names
    table.add_column("log₁₀(q)", justify="right", width=8)
    table.add_column("c", justify="right", width=6)
    table.add_column("ν", justify="right", width=4)
    table.add_column("φ", justify="right", width=6)
    # Unified-specific columns (February 2026)
    table.add_column("α", justify="right", width=6)  # alpha_asym
    table.add_column("γ", justify="right", width=5)  # gamma_vov
    table.add_column("BIC", justify="right", width=9)
    table.add_column("Hyv", justify="right", width=8)
    table.add_column("CRPS", justify="right", width=7)
    table.add_column("PIT p", justify="right", width=7)
    table.add_column("St", justify="center", width=3)
    
    last_group = None
    
    for asset, raw_data in sorted_assets:
            
        # Handle regime-conditional structure
        if 'global' in raw_data:
            data = raw_data['global']
        else:
            data = raw_data
        
        model = _model_label(raw_data)
        
        # Add group separator (13 columns total)
        if model != last_group:
            if last_group is not None:
                table.add_row("", "", "", "", "", "", "", "", "", "", "", "", "", style="dim")
            last_group = model
        
        q_val = data.get('q', float('nan'))
        c_val = data.get('c', 1.0)
        nu_val = data.get('nu')
        phi_val = data.get('phi')
        bic_val = data.get('bic', float('nan'))
        pit_p = data.get('pit_ks_pvalue', float('nan'))
        
        # Unified model specific parameters (February 2026)
        alpha_asym = data.get('alpha_asym')
        gamma_vov = data.get('gamma_vov')
        
        # Get Hyvarinen and CRPS from the best model in the models dict
        hyv_val = float('nan')
        crps_val = float('nan')
        models_dict = data.get('models', {})
        best_model_name = data.get('best_model_by_bic', data.get('noise_model', ''))
        if best_model_name and best_model_name in models_dict:
            best_model = models_dict[best_model_name]
            hyv_val = best_model.get('hyvarinen_score', float('nan'))
            crps_val = best_model.get('crps', float('nan'))
            if not np.isfinite(pit_p):
                pit_p = best_model.get('pit_ks_pvalue', float('nan'))
            # Get unified params from best model if not at top level
            if alpha_asym is None:
                alpha_asym = best_model.get('alpha_asym')
            if gamma_vov is None:
                gamma_vov = best_model.get('gamma_vov')
        elif models_dict:
            for m_name, m_data in models_dict.items():
                if m_data.get('fit_success', False):
                    hyv_val = m_data.get('hyvarinen_score', float('nan'))
                    crps_val = m_data.get('crps', float('nan'))
                    if not np.isfinite(pit_p):
                        pit_p = m_data.get('pit_ks_pvalue', float('nan'))
                    if alpha_asym is None:
                        alpha_asym = m_data.get('alpha_asym')
                    if gamma_vov is None:
                        gamma_vov = m_data.get('gamma_vov')
                    break
        
        log10_q = np.log10(q_val) if q_val > 0 else float('nan')
        
        # Format values
        q_str = f"{log10_q:.2f}" if np.isfinite(log10_q) else "-"
        c_str = f"{c_val:.3f}" if np.isfinite(c_val) else "-"
        nu_str = f"{nu_val:.0f}" if nu_val is not None else "-"
        phi_str = f"{phi_val:+.2f}" if phi_val is not None else "-"
        
        # Unified-specific columns
        alpha_str = f"{alpha_asym:+.2f}" if alpha_asym is not None else "[dim]-[/]"
        gamma_str = f"{gamma_vov:.2f}" if gamma_vov is not None else "[dim]-[/]"
        
        bic_str = f"{bic_val:.1f}" if np.isfinite(bic_val) else "-"
        
        # Hyvärinen score (higher is better, typically negative)
        if np.isfinite(hyv_val):
            hyv_str = f"{hyv_val:.1f}"
        else:
            hyv_str = "-"
        
        # CRPS (lower is better)
        if np.isfinite(crps_val):
            crps_str = f"{crps_val:.4f}"
        else:
            crps_str = "-"
        
        # PIT p-value with color coding
        if np.isfinite(pit_p):
            if pit_p >= 0.10:
                pit_str = f"[green]{pit_p:.4f}[/green]"
                status = "[green]✓[/green]"
            elif pit_p >= 0.05:
                pit_str = f"[yellow]{pit_p:.4f}[/yellow]"
                status = "[yellow]![/yellow]"
            else:
                pit_str = f"[red]{pit_p:.4f}[/red]"
                status = "[red]✗[/red]"
        else:
            pit_str = "-"
            status = "[dim]-[/dim]"
        
        # Model color based on type
        if "Unified" in model:
            model_style = "bright_yellow"
        elif "φ-t" in model or "Phi" in model.lower():
            model_style = "bright_magenta"
        elif "t(" in model:
            model_style = "magenta"
        elif "φ-Gauss" in model:
            model_style = "cyan"
        elif "Gauss" in model:
            model_style = "green"
        else:
            model_style = "white"
        
        table.add_row(
            asset,
            f"[{model_style}]{model}[/{model_style}]",
            q_str,
            c_str,
            nu_str,
            phi_str,
            alpha_str,
            gamma_str,
            beta_str,  # ELITE FIX: variance_inflation
            bic_str,
            hyv_str,
            crps_str,
            pit_str,
            status,
        )
    
    console.print(table)
    console.print()
    
    # Legend
    legend = Text()
    legend.append("    ", style="")
    legend.append("Legend: ", style="dim bold")
    legend.append("log₁₀(q)", style="dim")
    legend.append("=process noise  ", style="dim")
    legend.append("c", style="dim")
    legend.append("=obs scale  ", style="dim")
    legend.append("ν", style="dim")
    legend.append("=Student-t df  ", style="dim")
    legend.append("φ", style="dim")
    legend.append("=AR(1)  ", style="dim")
    legend.append("α", style="dim")
    legend.append("=asym  ", style="dim")
    legend.append("γ", style="dim")
    legend.append("=VoV  ", style="dim")
    legend.append("β", style="dim")
    legend.append("=var_infl  ", style="dim")
    legend.append("Hyv", style="dim")
    legend.append("=Hyvärinen  ", style="dim")
    legend.append("CRPS", style="dim")
    legend.append("=calib  ", style="dim")
    legend.append("PIT p", style="dim")
    legend.append("=p-val", style="dim")
    console.print(legend)
    console.print()


def render_dry_run_preview(assets: List[str], max_display: int = 20, console: Console = None) -> None:
    """Render dry run preview."""
    if console is None:
        console = create_tuning_console()
    console.print()
    console.print()
    warning_text = Text(justify="center")
    warning_text.append("\n⚠️  DRY RUN MODE\n", style="bold bright_yellow")
    warning_text.append("No changes will be made to cache\n", style="dim")
    warning_panel = Panel(Align.center(warning_text), box=box.ROUNDED, border_style="yellow", padding=(0, 4), width=45)
    console.print(Align.center(warning_panel))
    console.print()
    header = Text()
    header.append("  📋  ", style="bold bright_cyan")
    header.append(f"Would process {len(assets)} assets", style="white")
    console.print(header)
    console.print()


def render_failed_assets(failure_reasons: Dict[str, str], console: Console = None) -> None:
    """Render beautiful failed assets table with error categorization."""
    if console is None:
        console = create_tuning_console()
    if not failure_reasons:
        return
    
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    
    # Section header
    fail_section = Text()
    fail_section.append("  ❌  ", style="bold indian_red1")
    fail_section.append("FAILED ASSETS", style="bold indian_red1")
    fail_section.append(f"  ({len(failure_reasons)} assets)", style="dim")
    console.print(fail_section)
    console.print()
    
    # Categorize errors
    error_categories = {
        'data': [],      # Data fetch/availability issues
        'numeric': [],   # Numerical errors (NaN, convergence)
        'timeout': [],   # Timeout errors
        'api': [],       # API/rate limit errors
        'other': [],     # Other errors
    }
    
    for asset, reason in failure_reasons.items():
        reason_lower = reason.lower() if reason else ""
        
        # Categorize the error
        if any(x in reason_lower for x in ['no data', 'empty', 'missing', 'not found', 'unavailable', 'insufficient']):
            category = 'data'
            error_type = "Data unavailable"
        elif any(x in reason_lower for x in ['nan', 'inf', 'convergence', 'singular', 'numeric', 'overflow']):
            category = 'numeric'
            error_type = "Numeric error"
        elif any(x in reason_lower for x in ['timeout', 'timed out']):
            category = 'timeout'
            error_type = "Timeout"
        elif any(x in reason_lower for x in ['rate limit', 'api', '429', '403', 'forbidden']):
            category = 'api'
            error_type = "API error"
        else:
            category = 'other'
            error_type = "Error"
        
        # Extract first meaningful line of error
        first_line = reason.split('\n')[0][:55] if reason else "Unknown error"
        
        error_categories[category].append({
            'asset': asset,
            'error_type': error_type,
            'details': first_line,
            'full_reason': reason,
        })
    
    # Summary of error types
    summary = Text()
    summary.append("    ", style="")
    category_labels = [
        ('data', 'Data', 'yellow'),
        ('numeric', 'Numeric', 'bright_red'),
        ('timeout', 'Timeout', 'orange1'),
        ('api', 'API', 'bright_magenta'),
        ('other', 'Other', 'dim'),
    ]
    first = True
    for cat_key, cat_label, cat_color in category_labels:
        count = len(error_categories[cat_key])
        if count > 0:
            if not first:
                summary.append("   ·   ", style="dim")
            summary.append(f"{count}", style=f"bold {cat_color}")
            summary.append(f" {cat_label}", style="dim")
            first = False
    console.print(summary)
    console.print()
    
    # Create detailed table
    table = Table(
        show_header=True,
        header_style="bold white",
        border_style="indian_red1",
        box=box.ROUNDED,
        padding=(0, 1),
        row_styles=["", "on grey7"],
    )
    
    table.add_column("Asset", style="bold indian_red1", width=16, no_wrap=True)
    table.add_column("Category", justify="center", width=12)
    table.add_column("Error Details", style="dim", width=55, overflow="ellipsis")
    
    # Sort by category then by asset name
    all_failures = []
    for cat_key, cat_label, cat_color in category_labels:
        for item in sorted(error_categories[cat_key], key=lambda x: x['asset']):
            all_failures.append((item, cat_label, cat_color))
    
    # Display up to 30 failures
    for item, cat_label, cat_color in all_failures[:30]:
        table.add_row(
            item['asset'],
            f"[{cat_color}]{cat_label}[/{cat_color}]",
            item['details'],
        )
    
    console.print(table)
    
    # Show truncation notice if needed
    if len(all_failures) > 30:
        truncated = Text()
        truncated.append("    ... ", style="dim")
        truncated.append(f"{len(all_failures) - 30} more failures not shown", style="dim italic")
        console.print(truncated)
    
    console.print()
    
    # Actionable hint
    hint = Text()
    hint.append("    ", style="")
    hint.append("💡 ", style="bright_yellow")
    hint.append("Retry failed assets: ", style="dim")
    hint.append("make tune ARGS='--force --assets ", style="bright_cyan")
    # Show first few failed assets
    failed_list = list(failure_reasons.keys())[:5]
    hint.append(",".join(failed_list), style="bright_cyan")
    if len(failure_reasons) > 5:
        hint.append(",...", style="bright_cyan")
    hint.append("'", style="bright_cyan")
    console.print(hint)
    console.print()


def render_end_of_run_summary(
    processed_assets: Dict[str, Dict], regime_distributions: Dict[str, Dict[int, int]],
    model_comparisons: Dict[str, Dict], failure_reasons: Dict[str, str],
    processing_log: List[str], console: Console = None, cache: Dict = None
) -> None:
    """Render end-of-run summary."""
    if console is None:
        console = create_tuning_console()
    if failure_reasons:
        render_failed_assets(failure_reasons, console=console)
    if cache:
        render_calibration_report(cache, failure_reasons, console=console)


def render_calibration_report(cache: Dict, failure_reasons: Dict[str, str], console: Console = None) -> None:
    """Render calibration report showing assets with issues."""
    import numpy as np
    
    if console is None:
        console = create_tuning_console()
    
    issues = []
    
    # 1. Failed assets
    for asset, reason in (failure_reasons or {}).items():
        issues.append({
            'asset': asset,
            'issue_type': 'FAILED',
            'severity': 'critical',
            'pit_p': None,
            'ks_stat': None,
            'kurtosis': None,
            'model': '-',
            'q': None,
            'phi': None,
            'nu': None,
            'details': reason[:100] if reason else ''
        })
    
    # 2. Calibration warnings from cache
    for asset, raw_data in (cache or {}).items():
        if 'global' in raw_data:
            data = raw_data['global']
        else:
            data = raw_data
        
        pit_p = data.get('pit_ks_pvalue')
        ks_stat = data.get('ks_statistic')
        kurtosis = data.get('std_residual_kurtosis') or data.get('excess_kurtosis')
        calibration_warning = data.get('calibration_warning', False)
        noise_model = data.get('noise_model', '')
        q_val = data.get('q')
        phi_val = data.get('phi')
        nu_val = data.get('nu')
        
        # Check for ν refinement
        nu_refinement = data.get('nu_refinement') or {}
        nu_refinement_attempted = nu_refinement.get('refinement_attempted', False)
        nu_refinement_improved = nu_refinement.get('improvement_achieved', False)
        
        has_issue = False
        issue_type = []
        severity = 'ok'
        
        if calibration_warning or (pit_p is not None and pit_p < 0.05):
            has_issue = True
            if nu_refinement_attempted and nu_refinement_improved:
                issue_type.append('PIT < 0.05 (ν-ref)')
            elif nu_refinement_attempted:
                issue_type.append('PIT < 0.05 (ν-tried)')
            else:
                issue_type.append('PIT < 0.05')
            severity = 'warning'
        
        if pit_p is not None and pit_p < 0.01:
            severity = 'critical'
        
        if kurtosis is not None and kurtosis > 6:
            has_issue = True
            issue_type.append('High Kurt')
            if severity != 'critical':
                severity = 'warning'
        
        if has_issue:
            if 'student_t' in noise_model:
                model_str = f"φ-T(ν={int(nu_val)})" if nu_val else "Student-t"
            elif 'gaussian' in noise_model:
                model_str = "Gaussian"
            else:
                model_str = noise_model[:12] if noise_model else '-'
            
            issues.append({
                'asset': asset,
                'issue_type': ', '.join(issue_type),
                'severity': severity,
                'pit_p': pit_p,
                'ks_stat': ks_stat,
                'kurtosis': kurtosis,
                'model': model_str,
                'q': q_val,
                'phi': phi_val,
                'nu': nu_val,
                'details': '',
            })
    
    # Sort by severity (critical first), then by PIT p-value
    severity_order = {'critical': 0, 'warning': 1, 'ok': 2}
    issues.sort(key=lambda x: (severity_order.get(x['severity'], 2), x.get('pit_p') or 1.0))
    
    total_assets = len(cache) if cache else 0
    critical_count = sum(1 for i in issues if i['severity'] == 'critical')
    warning_count = sum(1 for i in issues if i['severity'] == 'warning')
    failed_count = sum(1 for i in issues if i['issue_type'] == 'FAILED')
    
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    
    # Section header
    section = Text()
    section.append("  ⚠️  ", style="bold bright_yellow")
    section.append("CALIBRATION ISSUES", style="bold bright_white")
    console.print(section)
    console.print()
    
    if not issues:
        success_text = Text()
        success_text.append("    ✓ ", style="bold bright_green")
        success_text.append("All ", style="white")
        success_text.append(f"{total_assets}", style="bold bright_cyan")
        success_text.append(" assets passed calibration checks", style="white")
        console.print(success_text)
        console.print()
        return
    
    # Summary stats
    summary = Text()
    summary.append("    ", style="")
    if critical_count > 0:
        summary.append(f"{critical_count}", style="bold indian_red1")
        summary.append(" critical", style="dim")
        summary.append("   ·   ", style="dim")
    if warning_count > 0:
        summary.append(f"{warning_count}", style="bold yellow")
        summary.append(" warnings", style="dim")
        summary.append("   ·   ", style="dim")
    if failed_count > 0:
        summary.append(f"{failed_count}", style="bold red")
        summary.append(" failed", style="dim")
        summary.append("   ·   ", style="dim")
    summary.append(f"{total_assets}", style="white")
    summary.append(" total assets", style="dim")
    console.print(summary)
    console.print()
    
    # Limit display to 50 worst issues
    display_issues = issues[:50]
    
    # Create issues table
    table = Table(
        show_header=True,
        header_style="bold white",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1),
        row_styles=["", "on grey7"],
    )
    
    table.add_column("Asset", justify="left", width=20, no_wrap=True)
    table.add_column("Issue", justify="left", width=18)
    table.add_column("PIT p", justify="right", width=8)
    table.add_column("KS", justify="right", width=6)
    table.add_column("Kurt", justify="right", width=6)
    table.add_column("Model", justify="left", width=12)
    table.add_column("log₁₀(q)", justify="right", width=9)
    table.add_column("φ", justify="right", width=6)
    
    for issue in display_issues:
        if issue['severity'] == 'critical':
            severity_style = "bold indian_red1"
            asset_style = "indian_red1"
        elif issue['severity'] == 'warning':
            severity_style = "yellow"
            asset_style = "yellow"
        else:
            severity_style = "dim"
            asset_style = "white"
        
        pit_str = f"{issue['pit_p']:.4f}" if issue['pit_p'] is not None else "-"
        ks_str = f"{issue['ks_stat']:.3f}" if issue['ks_stat'] is not None else "-"
        kurt_str = f"{issue['kurtosis']:.1f}" if issue['kurtosis'] is not None else "-"
        
        if issue['q'] is not None and issue['q'] > 0:
            log_q_str = f"{np.log10(issue['q']):.2f}"
        else:
            log_q_str = "-"
        
        phi_str = f"{issue['phi']:.3f}" if issue['phi'] is not None else "-"
        
        if issue['pit_p'] is not None:
            if issue['pit_p'] < 0.01:
                pit_styled = f"[bold indian_red1]{pit_str}[/]"
            elif issue['pit_p'] < 0.05:
                pit_styled = f"[yellow]{pit_str}[/]"
            else:
                pit_styled = f"[bright_green]{pit_str}[/]"
        else:
            pit_styled = "[dim]-[/]"
        
        if issue['kurtosis'] is not None:
            if issue['kurtosis'] > 10:
                kurt_styled = f"[bold indian_red1]{kurt_str}[/]"
            elif issue['kurtosis'] > 6:
                kurt_styled = f"[yellow]{kurt_str}[/]"
            else:
                kurt_styled = f"[dim]{kurt_str}[/]"
        else:
            kurt_styled = "[dim]-[/]"
        
        table.add_row(
            f"[{asset_style}]{issue['asset']}[/]",
            f"[{severity_style}]{issue['issue_type']}[/]",
            pit_styled,
            f"[dim]{ks_str}[/]",
            kurt_styled,
            f"[dim]{issue['model']}[/]",
            f"[dim]{log_q_str}[/]",
            f"[dim]{phi_str}[/]",
        )
    
    console.print(table)
    console.print()
    
    # Legend
    legend = Text()
    legend.append("    ", style="")
    legend.append("PIT p < 0.05", style="yellow")
    legend.append(" = model may be miscalibrated   ·   ", style="dim")
    legend.append("Kurt > 6", style="yellow")
    legend.append(" = heavy tails not fully captured", style="dim")
    console.print(legend)
    console.print()
    
    # PIT EXIT summary (February 2026)
    if PIT_PENALTY_AVAILABLE:
        render_pit_exit_summary(cache, console=console)


def render_pit_exit_summary(cache: Dict, console: Console = None) -> None:
    """
    Render summary of stocks that would trigger PIT EXIT signals.
    
    EXIT signals indicate that the model's belief cannot be trusted,
    requiring position closure and no new positions.
    """
    if console is None:
        console = create_tuning_console()
    
    if not PIT_PENALTY_AVAILABLE:
        return
    
    # Get critical stocks using strict threshold (p < 0.01 triggers EXIT)
    critical_stocks = get_pit_critical_stocks(cache, threshold=0.01)
    
    if not critical_stocks:
        return
    
    console.print()
    
    # Section header
    exit_header = Text()
    exit_header.append("  🚨 ", style="bold red")
    exit_header.append("PIT EXIT SIGNALS", style="bold red")
    exit_header.append(f"  ({len(critical_stocks)} stocks)", style="dim")
    console.print(exit_header)
    console.print()
    
    # Explanation
    explanation = Text()
    explanation.append("    ", style="")
    explanation.append("These stocks have critical PIT violations. ", style="dim italic")
    explanation.append("Signal = EXIT", style="bold red")
    explanation.append(" (close positions, no new trades).", style="dim italic")
    console.print(explanation)
    console.print()
    
    # List stocks in rows of 8
    for i in range(0, len(critical_stocks), 8):
        row_stocks = critical_stocks[i:i+8]
        row_text = Text()
        row_text.append("    ", style="")
        for j, stock in enumerate(row_stocks):
            if j > 0:
                row_text.append("  ", style="")
            row_text.append(stock, style="bold indian_red1")
        console.print(row_text)
    
    console.print()


class TuningProgressTracker:
    """Progress tracker for tuning with animated spinner."""

    def __init__(self, total_assets: int, console: Console = None):
        self.total = total_assets
        self.console = console or create_tuning_console()
        self.current = 0
        self.successes = 0
        self.failures = 0
        self.completed = []
        self.in_progress_assets = []
        self.progress = Progress(
            SpinnerColumn(spinner_name="dots", style="bright_yellow"),
            TextColumn("[bold cyan]{task.description}[/bold cyan]"),
            BarColumn(bar_width=30, complete_style="bright_green", finished_style="bright_green"),
            TaskProgressColumn(),
            TextColumn("·"),
            MofNCompleteColumn(),
            TextColumn("·"),
            TimeElapsedColumn(),
            console=self.console, transient=False, expand=False,
        )
        self.task_id = None
        self.progress.start()
        self.task_id = self.progress.add_task(description="Initializing...", total=total_assets)
    
    def set_in_progress(self, assets: list):
        self.in_progress_assets = list(assets) if assets else []
        self._update_description()
        self.progress.refresh()
    
    def _update_description(self):
        if self.in_progress_assets:
            shown = self.in_progress_assets[:4]
            desc = " · ".join(shown)
            if len(self.in_progress_assets) > 4:
                desc += f" (+{len(self.in_progress_assets) - 4})"
            self.progress.update(self.task_id, description=desc)
        elif self.current < self.total:
            self.progress.update(self.task_id, description="Processing...")
        else:
            self.progress.update(self.task_id, description="Complete")
    
    def add_in_progress(self, asset: str):
        if asset not in self.in_progress_assets:
            self.in_progress_assets.append(asset)
            self._update_description()
    
    def remove_in_progress(self, asset: str):
        if asset in self.in_progress_assets:
            self.in_progress_assets.remove(asset)
    
    def set_current(self, asset: str, model: str = ""):
        self.set_in_progress([asset])

    def update(self, asset: str, status: str, details: Optional[str] = None):
        self.current += 1
        self.remove_in_progress(asset)
        self._update_description()
        if status == 'success':
            self.successes += 1
            self.completed.append((asset, details, 'success'))
            model_short = self._extract_model_short(details)
            self.progress.console.print(f"  [green]✓[/green] [white]{asset}[/white] [dim]→[/dim] [bright_magenta]{model_short}[/bright_magenta]")
        elif status == 'failed':
            self.failures += 1
            error_first_line = details.split('\n')[0][:80] if details else "Error"
            self.completed.append((asset, error_first_line, 'failed'))
            self.progress.console.print()
            self.progress.console.print(f"  [bold red]✗ ERROR: {asset}[/bold red]")
            if details:
                for line in details.split('\n')[:8]:
                    self.progress.console.print(f"    [dim red]{line}[/dim red]")
            self.progress.console.print()
        self.progress.update(self.task_id, advance=1)
    
    def _extract_model_short(self, details: str) -> str:
        if not details:
            return ""
        parts = details.split('|')
        if parts:
            return parts[0][:35]
        return ""

    def finish(self):
        self.progress.stop()
        self.console.print()
        summary = Text()
        summary.append("  ▸ ", style="bright_green")
        summary.append(f"{self.successes}", style="bold green")
        summary.append(" tuned", style="dim")
        if self.failures > 0:
            summary.append("  ·  ", style="dim")
            summary.append(f"{self.failures}", style="bold red")
            summary.append(" failed", style="dim")
        self.console.print(summary)
        self.console.print()


class AuditAwareTuningProgressTracker(TuningProgressTracker):
    """Extended progress tracker with audit trail support."""
    
    def __init__(self, total_assets: int, console: Console = None):
        super().__init__(total_assets, console)
        self.audit_records = []
    
    def update(self, asset: str, status: str, details: Optional[str] = None, audit_record=None):
        if audit_record is not None:
            self.audit_records.append(audit_record)
        super().update(asset, status, details)
    
    def export_audit_trail(self):
        return [r.to_audit_dict() if hasattr(r, 'to_audit_dict') else r for r in self.audit_records]
    
    def get_escalation_summary(self):
        decisions = []
        for record in self.audit_records:
            if hasattr(record, 'escalation_decisions'):
                decisions.extend(record.escalation_decisions)
            elif isinstance(record, dict):
                decisions.extend(record.get('escalation_decisions', []))
        decision_counts = Counter(decisions)
        return {
            'total_records': len(self.audit_records),
            'decision_counts': dict(decision_counts),
            'escalation_rate': sum(1 for r in self.audit_records if self._has_escalation(r)) / max(len(self.audit_records), 1),
        }
    
    def _has_escalation(self, record):
        if hasattr(record, 'escalation_decisions'):
            decisions = record.escalation_decisions
        elif isinstance(record, dict):
            decisions = record.get('escalation_decisions', [])
        else:
            return False
        for d in decisions:
            if hasattr(d, 'name'):
                if d.name != 'HOLD_CURRENT':
                    return True
            elif isinstance(d, str):
                if d != 'hold_current' and d != 'HOLD_CURRENT':
                    return True
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Kalman MLE Tuning with World-Class UX",
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
    parser.add_argument('--cache-json', type=str, default='src/data/tune',
                       help='Path to cache directory (per-asset) or legacy JSON file')
    parser.add_argument('--force', action='store_true',
                       help='Force re-estimation even if cached values exist')
    parser.add_argument('--force-escalation', action='store_true',
                       help='Force re-estimation only for assets that failed calibration without escalation')
    parser.add_argument('--start', type=str, default='2015-01-01',
                       help='Start date for data fetching')
    parser.add_argument('--end', type=str, default=None,
                       help='End date for data fetching (default: today)')
    parser.add_argument('--max-assets', type=int, default=None,
                       help='Maximum number of assets to process (useful for testing)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what would be done without actually processing')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output (stack traces on errors)')
    parser.add_argument('--prior-mean', type=float, default=-6.0,
                       help='Prior mean for log10(q) (default: -6.0)')
    parser.add_argument('--prior-lambda', type=float, default=1.0,
                       help='Regularization strength (default: 1.0, set to 0 to disable)')
    parser.add_argument('--lambda-regime', type=float, default=0.05,
                       help='Hierarchical shrinkage toward global (default: 0.05, set to 0 for original behavior)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show all tuning output including per-model details (may clutter display)')

    args = parser.parse_args()

    # Enable debug mode
    if args.debug:
        os.environ['DEBUG'] = '1'
    
    # Enable verbose mode - unset TUNING_QUIET to show all messages
    if args.verbose:
        os.environ.pop('TUNING_QUIET', None)
        os.environ['TUNING_VERBOSE'] = '1'

    # Create console for rich output
    console = create_tuning_console()

    # Render beautiful header
    render_tuning_header(
        prior_mean=args.prior_mean,
        prior_lambda=args.prior_lambda,
        lambda_regime=args.lambda_regime,
        console=console,
    )

    # Load asset list
    assets = load_asset_list(args.assets, args.assets_file)

    # Apply max-assets limit
    if args.max_assets:
        assets = assets[:args.max_assets]

    # Dry-run mode
    if args.dry_run:
        render_dry_run_preview(assets, console=console)
        return

    # Load existing cache
    cache = load_cache(args.cache_json)

    # Process counters
    new_estimates = 0
    reused_cached = 0
    failed = 0
    calibration_warnings = 0
    student_t_count = 0
    gaussian_count = 0
    regime_tuning_count = 0
    # K=2 mixture removed (empirically falsified: 206 attempts, 0 selections)
    # Counters kept for backward compatibility with cached results
    mixture_attempted_count = 0
    mixture_selected_count = 0
    nu_refinement_attempted_count = 0
    nu_refinement_improved_count = 0
    gh_attempted_count = 0
    gh_selected_count = 0
    regime_tuning_count = 0
    
    # New model counters for comprehensive MODEL SELECTION display
    phi_gaussian_count = 0
    phi_student_t_count = 0
    phi_skew_t_count = 0
    phi_nig_count = 0
    gmm_fitted_count = 0
    hansen_fitted_count = 0
    hansen_left_skew_count = 0
    hansen_right_skew_count = 0
    evt_fitted_count = 0
    evt_heavy_tail_count = 0
    evt_moderate_tail_count = 0
    evt_light_tail_count = 0
    contaminated_t_count = 0
    tvvm_attempted_count = 0
    tvvm_selected_count = 0
    
    # Momentum augmentation counters (specific breakdown)
    momentum_count = 0
    momentum_gaussian_count = 0  # Gaussian with momentum (no phi)
    momentum_phi_gaussian_count = 0  # φ-Gaussian with momentum
    momentum_phi_student_t_count = 0  # φ-Student-t with momentum
    
    # Enhanced Student-t counters (February 2026)
    vov_enhanced_count = 0  # Vol-of-Vol enhanced
    two_piece_count = 0  # Two-Piece asymmetric tails
    mixture_t_count = 0  # Two-Component mixture
    unified_model_count = 0  # Unified Elite Student-t models
    
    # Volatility estimator counters (February 2026)
    gk_vol_count = 0  # Garman-Klass volatility
    har_vol_count = 0  # HAR (Corsi) volatility
    ewma_vol_count = 0  # EWMA volatility (fallback)
    
    # CRPS model selection counters (February 2026)
    crps_computed_count = 0  # Models with CRPS computed
    crps_regime_aware_count = 0  # Assets using regime-aware CRPS weights
    
    # Calibrated Trust Authority statistics
    recalibration_applied_count = 0
    calibrated_trust_count = 0
    trust_effective_values = []  # For computing average trust

    assets_to_process: List[str] = []
    failure_reasons: Dict[str, str] = {}
    
    # Comprehensive data collection for end-of-run summary
    processed_assets: Dict[str, Dict] = {}  # Full results per asset
    regime_distributions: Dict[str, Dict[int, int]] = {}  # Per-asset regime counts
    model_comparisons: Dict[str, Dict] = {}  # Per-asset model comparison results
    processing_log: List[str] = []  # Log of what was processed

    # Helper function to check if asset needs escalation re-tuning
    def needs_escalation_retune(data: Dict) -> bool:
        """Check if asset failed calibration without proper escalation attempt."""
        global_data = data.get('global', data)
        pit_p = global_data.get('pit_ks_pvalue', 1.0)
        calibration_warning = global_data.get('calibration_warning', False)
        mixture_attempted = global_data.get('mixture_attempted', False)
        nu_ref = global_data.get('nu_refinement') or {}
        nu_refinement_attempted = nu_ref.get('refinement_attempted', False) or global_data.get('nu_refinement_attempted', False)
        
        # Asset needs re-tuning if:
        # 1. Has calibration warning (PIT < 0.05)
        # 2. AND neither mixture nor ν-refinement was attempted
        if calibration_warning or pit_p < 0.05:
            if not mixture_attempted and not nu_refinement_attempted:
                return True
        return False

    # Check cache for each asset
    for asset in assets:
        if args.force:
            # Force mode: re-tune all
            assets_to_process.append(asset)
        elif args.force_escalation and asset in cache:
            # Force-escalation mode: only re-tune if escalation was skipped
            if needs_escalation_retune(cache[asset]):
                assets_to_process.append(asset)
            else:
                reused_cached += 1
        elif asset in cache:
            # Normal mode: use cached value
            reused_cached += 1
        else:
            # Asset not in cache: always process
            assets_to_process.append(asset)

    if assets_to_process:
        # Parallel processing
        import multiprocessing
        n_workers = multiprocessing.cpu_count()
        
        render_tuning_progress_start(
            len(assets_to_process), 
            n_workers, 
            reused_cached,
            len(cache),
            args.cache_json,
            console=console
        )
        
        # Create progress tracker with audit trail support (Counter-Proposal v1.0)
        # Use AuditAwareTuningProgressTracker to separate display from authoritative state
        tracker = AuditAwareTuningProgressTracker(len(assets_to_process), console=console)
        
        # Initialize escalation statistics tracker if control policy is available
        escalation_stats = EscalationStatistics() if CONTROL_POLICY_AVAILABLE else None

        # Prepare arguments for workers
        worker_args = []
        for asset in assets_to_process:
            prev_posteriors = _extract_previous_posteriors(cache.get(asset))
            worker_args.append(
                (asset, args.start, args.end, args.prior_mean, args.prior_lambda, args.lambda_regime, prev_posteriors)
            )

        # Process in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_tune_worker, arg): arg[0] for arg in worker_args}

            # Track in-flight assets for display
            in_flight = list(futures.values())
            
            # Show initial assets being processed
            # Use all assets since they're all submitted at once
            tracker.set_in_progress(in_flight)

            for future in as_completed(futures):
                asset = futures[future]
                if asset in in_flight:
                    in_flight.remove(asset)
                
                # Update in-progress list as jobs complete
                tracker.set_in_progress(in_flight)
                
                try:
                    asset_name, result, error, traceback_str = future.result()

                    if result:
                        cache[asset_name] = result
                        new_estimates += 1
                        regime_tuning_count += 1
                        
                        # Store full result for end-of-run summary
                        processed_assets[asset_name] = result

                        global_result = result.get('global', result)
                        noise_model = global_result.get('noise_model', '')
                        
                        # Count base distribution models
                        if noise_model.startswith('phi_nig_'):
                            phi_nig_count += 1
                            student_t_count += 1  # Also count as heavy-tailed
                        elif noise_model.startswith('phi_skew_t_nu_'):
                            phi_skew_t_count += 1
                            student_t_count += 1  # Also count as heavy-tailed
                        elif noise_model.startswith('phi_student_t_nu_'):
                            phi_student_t_count += 1
                            student_t_count += 1  # Count all heavy-tailed models together
                        elif noise_model == 'phi_gaussian' or 'phi' in noise_model.lower():
                            phi_gaussian_count += 1
                            gaussian_count += 1
                        else:
                            gaussian_count += 1
                        
                        # Count momentum augmentation
                        best_model = global_result.get('best_model', '')
                        is_momentum = global_result.get('is_momentum_model', False) or '_momentum' in str(best_model)
                        if is_momentum:
                            momentum_count += 1

                        if global_result.get('calibration_warning'):
                            calibration_warnings += 1
                        
                        # Track K=2 mixture model attempts and selections
                        if global_result.get('mixture_attempted'):
                            mixture_attempted_count += 1
                        if global_result.get('mixture_selected'):
                            mixture_selected_count += 1
                        
                        # Track adaptive ν refinement attempts and improvements
                        nu_refinement = global_result.get('nu_refinement') or {}
                        if nu_refinement.get('refinement_attempted') or global_result.get('nu_refinement_attempted'):
                            nu_refinement_attempted_count += 1
                        if nu_refinement.get('improvement_achieved') or global_result.get('nu_refinement_improved'):
                            nu_refinement_improved_count += 1
                        
                        # Track GH distribution attempts and selections
                        if global_result.get('gh_attempted'):
                            gh_attempted_count += 1
                        if global_result.get('gh_selected'):
                            gh_selected_count += 1
                        
                        # Collect regime distribution
                        if result.get('regime_counts'):
                            regime_distributions[asset_name] = result['regime_counts']
                        
                        # Collect model comparison data
                        if global_result.get('model_comparison'):
                            model_comparisons[asset_name] = {
                                'model_comparison': global_result['model_comparison'],
                                'selected_model': global_result.get('noise_model', 'unknown'),
                                'best_model': global_result.get('best_model', global_result.get('best_model_by_bic', 'unknown')),
                                'q': global_result.get('q'),
                                'c': global_result.get('c'),
                                'phi': global_result.get('phi'),
                                'nu': global_result.get('nu'),
                                'bic': global_result.get('bic'),
                                'aic': global_result.get('aic'),
                                'log_likelihood': global_result.get('log_likelihood'),
                                'n_obs': global_result.get('n_obs'),
                            }

                        # Update progress tracker with rich model info
                        q_val = global_result.get('q', float('nan'))
                        c_val = global_result.get('c', 1.0)
                        phi_val = global_result.get('phi')
                        nu_val = global_result.get('nu')
                        bic_val = global_result.get('bic', float('nan'))
                        model_type = global_result.get('noise_model', 'gaussian')
                        nu_was_refined = nu_refinement.get('improvement_achieved', False) or global_result.get('nu_refinement_improved', False)
                        
                        # Build comprehensive details string for UX display
                        # Format: model|q|c|phi|nu|bic|trust
                        if global_result.get('gh_selected'):
                            gh_model = global_result.get('gh_model', {})
                            gh_params = gh_model.get('parameters', {})
                            beta = gh_params.get('beta', 0)
                            skew = gh_model.get('skewness_direction', 'sym')[:1].upper()
                            model_str = f"GH(β={beta:.1f},{skew})"
                        elif global_result.get('mixture_selected'):
                            mixture_model = global_result.get('mixture_model', {})
                            sigma_ratio = mixture_model.get('sigma_ratio', 0)
                            weight = mixture_model.get('weight', 0)
                            model_str = f"K2-Mix(σ={sigma_ratio:.1f})"
                        elif model_type.startswith('phi_nig_'):
                            # φ-NIG model with alpha/beta parameters
                            nig_alpha = global_result.get('nig_alpha')
                            nig_beta = global_result.get('nig_beta')
                            if nig_beta is not None and abs(nig_beta) > 0.01:
                                skew_dir = "L" if nig_beta < 0 else "R"
                                model_str = f"NIG({skew_dir})"
                            else:
                                model_str = "NIG"
                        elif model_type.startswith('phi_skew_t_nu_'):
                            # φ-Skew-t model with gamma parameter
                            gamma_val = global_result.get('gamma')
                            if gamma_val is not None and abs(gamma_val - 1.0) > 0.01:
                                skew_dir = "L" if gamma_val < 1.0 else "R"
                                model_str = f"Skew-t({skew_dir})"
                            else:
                                model_str = "Skew-t"
                        elif global_result.get('unified_model') or (model_type and 'unified' in str(model_type).lower()):
                            # UNIFIED Elite Student-t Model (February 2026)
                            alpha_asym = global_result.get('alpha_asym', 0)
                            gamma_vov_u = global_result.get('gamma_vov', 0)
                            ms_sens = global_result.get('ms_sensitivity', 2.0)
                            degraded = global_result.get('degraded', False)
                            
                            # Build unified model string with ν
                            nu_str = f"ν={int(nu_val)}" if nu_val else ""
                            model_str = f"φ-t-Uni({nu_str})"
                            
                            # Add enhancement indicators
                            enhancements = []
                            if alpha_asym and abs(alpha_asym) > 0.01:
                                skew_dir = "L" if alpha_asym < 0 else "R"
                                enhancements.append(f"α{skew_dir}")
                            if gamma_vov_u and gamma_vov_u > 0.05:
                                enhancements.append("VoV")
                            if ms_sens and abs(ms_sens - 2.0) > 0.1:
                                enhancements.append("MS-q")
                            
                            if enhancements:
                                model_str += "+" + "+".join(enhancements)
                            
                            if degraded:
                                model_str += "⚠"
                            
                            unified_model_count += 1
                        elif model_type.startswith('phi_student_t_nu_') and nu_val is not None:
                            # Check for Enhanced Student-t variants
                            gamma_vov = global_result.get('gamma_vov')
                            nu_left = global_result.get('nu_left')
                            nu_right = global_result.get('nu_right')
                            nu_calm = global_result.get('nu_calm')
                            nu_stress = global_result.get('nu_stress')
                            
                            if gamma_vov is not None and gamma_vov > 0:
                                model_str = "Student-t+VoV"
                            elif nu_left is not None and nu_right is not None:
                                model_str = "Student-t+2P"
                            elif nu_calm is not None and nu_stress is not None:
                                model_str = "Student-t+Mix"
                            else:
                                model_str = "Student-t"
                        elif phi_val is not None:
                            model_str = "φ-Gaussian"
                        else:
                            model_str = "Gaussian"
                        
                        # Check for momentum augmentation
                        is_momentum_model = global_result.get('is_momentum_model', False)
                        if is_momentum_model or (model_type and '_momentum' in str(model_type)):
                            model_str += "+Momentum"
                        
                        # Check for GMM availability
                        gmm_data = global_result.get('gmm')
                        has_gmm = (gmm_data is not None and 
                                   isinstance(gmm_data, dict) and 
                                   not gmm_data.get('is_degenerate', False))
                        if has_gmm:
                            model_str += "+GMM"
                            gmm_fitted_count += 1
                        
                        # Check for Hansen Skew-t availability
                        hansen_data = global_result.get('hansen_skew_t')
                        has_hansen = (hansen_data is not None and 
                                      isinstance(hansen_data, dict) and
                                      hansen_data.get('lambda') is not None and
                                      abs(hansen_data.get('lambda', 0)) > 0.01)
                        if has_hansen:
                            hansen_lambda = hansen_data.get('lambda', 0)
                            hansen_dir = "←" if hansen_lambda < 0 else "→"
                            model_str += f"+Hλ{hansen_dir}"
                            hansen_fitted_count += 1
                        
                        # Check for EVT availability
                        evt_data = global_result.get('evt')
                        has_evt = (evt_data is not None and 
                                   isinstance(evt_data, dict) and
                                   evt_data.get('fit_success', False))
                        if has_evt:
                            evt_xi = evt_data.get('xi', 0)
                            tail_type = "H" if evt_xi > 0.2 else ("M" if evt_xi > 0.05 else "L")
                            model_str += f"+EVT{tail_type}"
                            evt_fitted_count += 1
                            if evt_xi > 0.2:
                                evt_heavy_tail_count += 1
                            elif evt_xi > 0.05:
                                evt_moderate_tail_count += 1
                            else:
                                evt_light_tail_count += 1
                        
                        # Check for Contaminated Student-t availability
                        cst_data = global_result.get('contaminated_student_t')
                        has_cst = (cst_data is not None and 
                                   isinstance(cst_data, dict) and
                                   cst_data.get('nu_normal') is not None and
                                   cst_data.get('nu_crisis') is not None)
                        if has_cst:
                            cst_epsilon = cst_data.get('epsilon', 0.05)
                            model_str += f"+CST{int(cst_epsilon*100)}%"
                            contaminated_t_count += 1
                        
                        # Count volatility estimator (February 2026)
                        vol_estimator = global_result.get('volatility_estimator', 'EWMA')
                        if vol_estimator == 'GK' or vol_estimator == 'Garman-Klass':
                            gk_vol_count += 1
                        elif vol_estimator == 'HAR' or vol_estimator == 'HAR-GK':
                            har_vol_count += 1
                        else:
                            ewma_vol_count += 1
                        
                        import math
                        details = f"{model_str}|q={q_val:.2e}|c={c_val:.3f}"
                        if phi_val is not None:
                            details += f"|φ={phi_val:+.2f}"
                        if nu_val is not None:
                            nu_indicator = f"ν={int(nu_val)}" + ("*" if nu_was_refined else "")
                            details += f"|{nu_indicator}"
                        # Add gamma for skew-t models
                        gamma_val = global_result.get('gamma')
                        if gamma_val is not None and abs(gamma_val - 1.0) > 0.01:
                            details += f"|γ={gamma_val:.2f}"
                        # Add Hansen lambda for asymmetric tails
                        if has_hansen:
                            details += f"|λ={hansen_lambda:+.2f}"
                        # Add EVT xi for tail heaviness
                        if has_evt:
                            details += f"|ξ={evt_xi:.2f}"
                        # Add Contaminated Student-t crisis ν
                        if has_cst:
                            cst_nu_normal = cst_data.get('nu_normal', 12)
                            cst_nu_crisis = cst_data.get('nu_crisis', 4)
                            details += f"|ν_c={int(cst_nu_crisis)}"
                        if math.isfinite(bic_val):
                            details += f"|bic={bic_val:.0f}"
                        
                        # Add trust indicator if available
                        effective_trust = global_result.get('effective_trust')
                        if effective_trust is not None:
                            trust_pct = effective_trust * 100
                            if trust_pct >= 70:
                                trust_indicator = f"T={trust_pct:.0f}%✓"
                            elif trust_pct < 30:
                                trust_indicator = f"T={trust_pct:.0f}%⚠"
                            else:
                                trust_indicator = f"T={trust_pct:.0f}%"
                            details += f"|{trust_indicator}"
                        
                        # Log this processing for end-of-run
                        processing_log.append(f"✓ {asset_name}: {details}")
                        
                        tracker.update(asset_name, 'success', details)
                    else:
                        failed += 1
                        error_msg = error or "tuning returned None"
                        # Build full error with traceback
                        full_error = error_msg
                        if traceback_str:
                            full_error = f"{error_msg}\n{traceback_str}"
                        failure_reasons[asset_name] = full_error
                        processing_log.append(f"❌ {asset_name}: {error_msg}")
                        # Pass full error to tracker so it's displayed
                        tracker.update(asset_name, 'failed', full_error)

                except Exception as e:
                    import traceback
                    failed += 1
                    tb_str = traceback.format_exc()
                    full_error = f"{str(e)}\n{tb_str}"
                    failure_reasons[asset] = full_error
                    processing_log.append(f"❌ {asset}: {str(e)}")
                    # Pass full error to tracker so it's displayed
                    tracker.update(asset, 'failed', full_error)
        
        tracker.finish()
    else:
        # All cached - show minimal info
        from rich.align import Align
        from rich.text import Text
        console.print()
        info = Text()
        info.append("○", style="dim cyan")
        info.append(f"  All {len(assets)} assets cached", style="dim")
        info.append(f"  ·  {len(cache):,} total in cache", style="dim")
        console.print(Align.center(info))
        console.print()

    # ========================================================================
    # MERGE NEWLY TUNED ASSETS INTO CACHE BEFORE SAVING
    # ========================================================================
    # CRITICAL: processed_assets contains all newly tuned results.
    # These must be merged into cache before saving, otherwise new results
    # are lost and only old cached values persist.
    # ========================================================================
    if processed_assets:
        cache.update(processed_assets)

    # Save updated cache
    if new_estimates > 0:
        save_cache_json(cache, args.cache_json)

    # Count regime statistics from cache
    regime_fit_counts = {r: 0 for r in range(5)}
    regime_shrunk_counts = {r: 0 for r in range(5)}
    collapse_warnings = 0
    
    # Detailed model breakdown: {regime_id: {'gaussian': count, 'phi_gaussian': count, 'student_t_4': count, ...}}
    regime_model_breakdown = {r: {} for r in range(5)}
    
    for asset, data in cache.items():
        # Get the global noise model for this asset
        global_data = data.get('global', data)
        noise_model = global_data.get('noise_model', 'gaussian') or 'gaussian'
        nu_val = global_data.get('nu')
        phi_val = global_data.get('phi')
        gamma_val = global_data.get('gamma')
        nig_alpha = global_data.get('nig_alpha')
        nig_beta = global_data.get('nig_beta')
        mixture_selected = global_data.get('mixture_selected', False)
        mixture_model = global_data.get('mixture_model', {})
        
        # Get augmentation layer data
        gmm_data = global_data.get('gmm')
        hansen_data = global_data.get('hansen_skew_t')
        evt_data = global_data.get('evt')
        cst_data = global_data.get('contaminated_student_t')
        
        # Detect momentum and strip suffix for base model determination
        is_momentum_model = '_momentum' in noise_model
        base_noise_model = noise_model.replace('_momentum', '')
        
        # Determine base model category
        if mixture_selected and mixture_model:
            sigma_ratio = mixture_model.get('sigma_ratio', 0)
            model_key = f"K2-Mix(σ={sigma_ratio:.1f})"
        elif base_noise_model.startswith('phi_nig_'):
            # φ-NIG model with alpha/beta parameters
            if nig_beta is not None and abs(nig_beta) > 0.01:
                skew_dir = "L" if nig_beta < 0 else "R"
                model_key = f"φ-NIG({skew_dir})"
            else:
                model_key = "φ-NIG"
        elif base_noise_model.startswith('phi_skew_t_nu_'):
            # φ-Skew-t model with gamma parameter
            gamma_val = global_data.get('gamma')
            if gamma_val is not None and abs(gamma_val - 1.0) > 0.01:
                skew_dir = "L" if gamma_val < 1.0 else "R"
                model_key = f"φ-Skew-t({skew_dir})"
            else:
                model_key = "φ-Skew-t"
        elif base_noise_model.startswith('phi_student_t_nu_') and nu_val is not None:
            model_key = f"φ-t(ν={int(nu_val)})"
        elif base_noise_model == 'kalman_phi_gaussian' or phi_val is not None:
            model_key = "φ-Gaussian"
        else:
            model_key = "Gaussian"
        
        # Create augmentation suffix for tracking
        aug_suffix = ""
        if gmm_data is not None and isinstance(gmm_data, dict) and not gmm_data.get('is_degenerate', False):
            aug_suffix += "+GMM"
        if hansen_data is not None and isinstance(hansen_data, dict):
            hansen_lambda = hansen_data.get('lambda')
            if hansen_lambda is not None and abs(hansen_lambda) > 0.01:
                aug_suffix += "+Hλ"
        if evt_data is not None and isinstance(evt_data, dict) and evt_data.get('fit_success', False):
            aug_suffix += "+EVT"
        if cst_data is not None and isinstance(cst_data, dict) and cst_data.get('nu_normal') is not None:
            aug_suffix += "+CST"
        
        # Store both base model and augmented model for breakdown
        regime_data = data.get('regime')
        if regime_data is not None and isinstance(regime_data, dict):
            for r, params in regime_data.items():
                if isinstance(params, dict):
                    is_fallback = params.get('fallback', False) or params.get('regime_meta', {}).get('fallback', False)
                    r_int = int(r)
                    
                    if not is_fallback:
                        regime_fit_counts[r_int] += 1
                        
                        # Count model breakdown per regime
                        # Use EITHER base model key OR momentum model key (not both) for cleaner accounting
                        if is_momentum_model:
                            # Use momentum model key instead of base model key
                            if model_key.startswith("φ-t(ν=") or "Student" in model_key:
                                effective_key = "φ-Student-t+Mom"
                            elif model_key == "φ-Gaussian":
                                effective_key = "φ-Gaussian+Mom"
                            elif model_key == "Gaussian":
                                effective_key = "Gaussian+Mom"
                            else:
                                effective_key = "Gaussian+Mom"
                        else:
                            # Use base model key for non-momentum models
                            effective_key = model_key
                        
                        if effective_key not in regime_model_breakdown[r_int]:
                            regime_model_breakdown[r_int][effective_key] = 0
                        regime_model_breakdown[r_int][effective_key] += 1
                        
                        is_shrunk = params.get('shrinkage_applied', False) or params.get('regime_meta', {}).get('shrinkage_applied', False)
                        if is_shrunk:
                            regime_shrunk_counts[r_int] += 1
                    
                    # ==================================================================
                    # AUGMENTATION LAYERS: Count per regime (global layers apply to all)
                    # ==================================================================
                    # These are fitted at the global level but apply to regime-specific
                    # models, so count them for each regime that has any fit (fallback or not)
                    # This ensures proper display in REGIME COVERAGE table
                    # ==================================================================
                    if gmm_data is not None and isinstance(gmm_data, dict) and not gmm_data.get('is_degenerate', False):
                        if "GMM" not in regime_model_breakdown[r_int]:
                            regime_model_breakdown[r_int]["GMM"] = 0
                        regime_model_breakdown[r_int]["GMM"] += 1
                    
                    if hansen_data is not None and isinstance(hansen_data, dict):
                        hansen_lambda = hansen_data.get('lambda')
                        if hansen_lambda is not None and abs(hansen_lambda) > 0.01:
                            if "Hansen-λ" not in regime_model_breakdown[r_int]:
                                regime_model_breakdown[r_int]["Hansen-λ"] = 0
                            regime_model_breakdown[r_int]["Hansen-λ"] += 1
                    
                    if evt_data is not None and isinstance(evt_data, dict) and evt_data.get('fit_success', False):
                        if "EVT" not in regime_model_breakdown[r_int]:
                            regime_model_breakdown[r_int]["EVT"] = 0
                        regime_model_breakdown[r_int]["EVT"] += 1
                    
                    if cst_data is not None and isinstance(cst_data, dict) and cst_data.get('nu_normal') is not None:
                        if "CST" not in regime_model_breakdown[r_int]:
                            regime_model_breakdown[r_int]["CST"] = 0
                        regime_model_breakdown[r_int]["CST"] += 1
                    
        if 'hierarchical_tuning' in data:
            if data['hierarchical_tuning'].get('collapse_warning', False):
                collapse_warnings += 1

    # Compute escalation statistics from cache (for both fresh and cached runs)
    # These need to be computed from the full cache to show accurate totals
    mixture_attempted_count = 0
    mixture_selected_count = 0
    nu_refinement_attempted_count = 0
    nu_refinement_improved_count = 0
    gh_attempted_count = 0
    gh_selected_count = 0
    tvvm_attempted_count = 0
    tvvm_selected_count = 0
    calibration_warnings = 0
    gaussian_count = 0
    student_t_count = 0
    # Reset new model counters for full cache computation
    phi_gaussian_count = 0
    phi_student_t_count = 0
    phi_skew_t_count = 0
    phi_nig_count = 0
    gmm_fitted_count = 0
    hansen_fitted_count = 0
    hansen_left_skew_count = 0
    hansen_right_skew_count = 0
    evt_fitted_count = 0
    evt_heavy_tail_count = 0
    evt_moderate_tail_count = 0
    evt_light_tail_count = 0
    contaminated_t_count = 0
    # Reset momentum counters for full cache computation
    momentum_count = 0
    momentum_gaussian_count = 0
    momentum_student_t_count = 0
    # Reset trust statistics for full cache computation
    recalibration_applied_count = 0
    calibrated_trust_count = 0
    trust_effective_values = []
    
    for asset, data in cache.items():
        global_data = data.get('global', data)
        
        # Count model types
        noise_model = global_data.get('noise_model', '') or ''
        
        # Detect if this is a momentum model (noise_model ends with _momentum)
        is_momentum = '_momentum' in noise_model
        
        # Count by model type - handle momentum suffix properly
        if noise_model.startswith('phi_nig_'):
            phi_nig_count += 1
            student_t_count += 1
        elif noise_model.startswith('phi_skew_t_nu_'):
            phi_skew_t_count += 1
            student_t_count += 1
        elif 'phi_student_t_nu_' in noise_model:
            # phi_student_t_nu_8 or phi_student_t_nu_8_momentum
            phi_student_t_count += 1
            student_t_count += 1
            if is_momentum:
                momentum_phi_student_t_count += 1
                momentum_count += 1
        elif noise_model == 'phi_gaussian' or 'phi' in noise_model.lower():
            phi_gaussian_count += 1
            gaussian_count += 1
            if is_momentum:
                momentum_phi_gaussian_count += 1
                momentum_count += 1
        elif 'gaussian' in noise_model.lower():
            # kalman_gaussian or kalman_gaussian_momentum
            gaussian_count += 1
            if is_momentum:
                momentum_gaussian_count += 1
                momentum_count += 1
        
        # Count augmentation layers from cache
        gmm_data = global_data.get('gmm')
        if gmm_data is not None and isinstance(gmm_data, dict) and not gmm_data.get('is_degenerate', False):
            gmm_fitted_count += 1
        
        hansen_data = global_data.get('hansen_skew_t')
        if hansen_data is not None and isinstance(hansen_data, dict):
            hansen_lambda = hansen_data.get('lambda')
            if hansen_lambda is not None and abs(hansen_lambda) > 0.01:
                hansen_fitted_count += 1
                if hansen_lambda < 0:
                    hansen_left_skew_count += 1
                else:
                    hansen_right_skew_count += 1
        
        evt_data = global_data.get('evt')
        if evt_data is not None and isinstance(evt_data, dict) and evt_data.get('fit_success', False):
            evt_fitted_count += 1
            evt_xi = evt_data.get('xi', 0)
            if evt_xi > 0.2:
                evt_heavy_tail_count += 1
            elif evt_xi > 0.05:
                evt_moderate_tail_count += 1
            else:
                evt_light_tail_count += 1
        
        cst_data = global_data.get('contaminated_student_t')
        if cst_data is not None and isinstance(cst_data, dict):
            if cst_data.get('nu_normal') is not None and cst_data.get('nu_crisis') is not None:
                contaminated_t_count += 1
        
        # Note: Momentum counting is done above in the model classification section
        
        # Count calibration warnings
        if global_data.get('calibration_warning'):
            calibration_warnings += 1
        
        # Count mixture attempts and selections
        if global_data.get('mixture_attempted'):
            mixture_attempted_count += 1
        if global_data.get('mixture_selected'):
            mixture_selected_count += 1
        
        # Count ν refinement attempts and improvements
        nu_refinement = global_data.get('nu_refinement') or {}
        if nu_refinement.get('refinement_attempted') or global_data.get('nu_refinement_attempted'):
            nu_refinement_attempted_count += 1
        if nu_refinement.get('improvement_achieved') or global_data.get('nu_refinement_improved'):
            nu_refinement_improved_count += 1
        
        # Count GH attempts and selections
        if global_data.get('gh_attempted'):
            gh_attempted_count += 1
        if global_data.get('gh_selected'):
            gh_selected_count += 1
        
        # Count TVVM attempts and selections
        if global_data.get('tvvm_attempted'):
            tvvm_attempted_count += 1
        if global_data.get('tvvm_selected'):
            tvvm_selected_count += 1
        
        # Count Calibrated Trust Authority statistics
        if global_data.get('recalibration_applied'):
            recalibration_applied_count += 1
        if global_data.get('calibrated_trust'):
            calibrated_trust_count += 1
            effective_trust = global_data.get('effective_trust')
            if effective_trust is not None:
                trust_effective_values.append(effective_trust)

    # Compute trust statistics
    avg_effective_trust = sum(trust_effective_values) / len(trust_effective_values) if trust_effective_values else 0.0
    low_trust_count = sum(1 for t in trust_effective_values if t < 0.3)
    high_trust_count = sum(1 for t in trust_effective_values if t >= 0.7)

    # Render beautiful summary
    render_tuning_summary(
        total_assets=len(assets),
        new_estimates=new_estimates,
        reused_cached=reused_cached,
        failed=failed,
        calibration_warnings=calibration_warnings,
        gaussian_count=gaussian_count,
        student_t_count=student_t_count,
        regime_tuning_count=regime_tuning_count,
        lambda_regime=args.lambda_regime,
        regime_fit_counts=regime_fit_counts,
        regime_shrunk_counts=regime_shrunk_counts,
        collapse_warnings=collapse_warnings,
        cache_path=args.cache_json,
        regime_model_breakdown=regime_model_breakdown,
        mixture_attempted_count=mixture_attempted_count,
        mixture_selected_count=mixture_selected_count,
        nu_refinement_attempted_count=nu_refinement_attempted_count,
        nu_refinement_improved_count=nu_refinement_improved_count,
        gh_attempted_count=gh_attempted_count,
        gh_selected_count=gh_selected_count,
        tvvm_attempted_count=tvvm_attempted_count,
        tvvm_selected_count=tvvm_selected_count,
        # New model counters
        phi_gaussian_count=phi_gaussian_count,
        phi_student_t_count=phi_student_t_count,
        phi_skew_t_count=phi_skew_t_count,
        phi_nig_count=phi_nig_count,
        gmm_fitted_count=gmm_fitted_count,
        hansen_fitted_count=hansen_fitted_count,
        hansen_left_skew_count=hansen_left_skew_count,
        hansen_right_skew_count=hansen_right_skew_count,
        evt_fitted_count=evt_fitted_count,
        evt_heavy_tail_count=evt_heavy_tail_count,
        evt_moderate_tail_count=evt_moderate_tail_count,
        evt_light_tail_count=evt_light_tail_count,
        contaminated_t_count=contaminated_t_count,
        # Calibrated Trust Authority statistics
        recalibration_applied_count=recalibration_applied_count,
        calibrated_trust_count=calibrated_trust_count,
        avg_effective_trust=avg_effective_trust,
        low_trust_count=low_trust_count,
        high_trust_count=high_trust_count,
        # Momentum augmentation counts (specific breakdown)
        momentum_gaussian_count=momentum_gaussian_count,
        momentum_phi_gaussian_count=momentum_phi_gaussian_count,
        momentum_phi_student_t_count=momentum_phi_student_t_count,
        momentum_total_count=momentum_count,
        # Enhanced Student-t counts (February 2026)
        vov_enhanced_count=vov_enhanced_count,
        two_piece_count=two_piece_count,
        mixture_t_count=mixture_t_count,
        # Volatility estimator counts (February 2026)
        gk_vol_count=gk_vol_count,
        har_vol_count=har_vol_count,
        ewma_vol_count=ewma_vol_count,
        # CRPS model selection (February 2026)
        crps_computed_count=crps_computed_count,
        crps_regime_aware_count=crps_regime_aware_count,
        console=console,
    )

    # Render Elite Tuning diagnostics summary (plateau-optimal parameter selection)
    render_elite_tuning_summary(cache, console=console)

    # Render PDDE escalation summary if available
    if PDDE_AVAILABLE and cache:
        try:
            escalation_summary = get_escalation_summary_from_cache(cache)
            if escalation_summary.get('total', 0) > 0:
                render_pdde_escalation_summary(escalation_summary, console=console)
        except Exception:
            pass  # Silently skip if PDDE summary fails

    # Render Market Risk Temperature with Crash Risk Assessment
    try:
        from decision.risk_temperature import get_cached_risk_temperature
        from decision.signals_ux import render_risk_temperature_summary
        
        risk_temp_result = get_cached_risk_temperature(
            start_date="2020-01-01",
            notional=1_000_000,
            estimated_gap_risk=0.03,
        )
        render_risk_temperature_summary(risk_temp_result, console=console)
    except Exception:
        pass  # Silently skip if risk temperature fails

    # Render parameter table
    if cache:
        render_parameter_table(cache, console=console)

    # Render comprehensive end-of-run summary with all collected data
    render_end_of_run_summary(
        processed_assets=processed_assets,
        regime_distributions=regime_distributions,
        model_comparisons=model_comparisons,
        failure_reasons=failure_reasons,
        processing_log=processing_log,
        console=console,
        cache=cache,
    )


if __name__ == '__main__':
    main()
