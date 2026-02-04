#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
UNIFIED RISK DASHBOARD
═══════════════════════════════════════════════════════════════════════════════

Combines all risk temperature modules into a single unified view:
  - Risk Temperature (cross-asset stress indicators)
  - Metals Risk Temperature (precious/industrial metals)
  - Market Temperature (US equity market assessment)

Invoked via: make risk

Design Philosophy:
  "One dashboard to rule them all"
  
  A senior portfolio manager needs a single view of:
    1. Overall market stress level
    2. Metals-specific crash risk indicators
    3. Equity market momentum and sector rotation
    4. Position sizing recommendations

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Suppress noisy loggers
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# Add src to path
SCRIPT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.align import Align
from rich.rule import Rule


# =============================================================================
# OUTPUT SUPPRESSION CONTEXT MANAGER
# =============================================================================

class _SuppressOutput:
    """Context manager to suppress stdout/stderr (for noisy yfinance)."""
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        return False


# =============================================================================
# ACTIONABLE GUIDANCE HELPERS
# =============================================================================

def _render_cross_asset_guidance(console: Console, risk_result) -> None:
    """Render actionable guidance for Cross-Asset stress indicators."""
    categories = getattr(risk_result, 'categories', {})
    if not categories:
        return
    
    high_stress = []
    for cat_key, cat in categories.items():
        stress = getattr(cat, 'stress_level', 0.0)
        if stress >= 1.5:
            high_stress.append((cat_key, stress, "CRITICAL"))
        elif stress >= 1.0:
            high_stress.append((cat_key, stress, "HIGH"))
    
    if not high_stress:
        console.print("  [dim]📊 Guidance:[/dim] [bright_green]All stress indicators normal.[/bright_green]")
        console.print("     [dim]→ Full position sizing permitted. Monitor for changes.[/dim]")
        return
    
    console.print("  [dim]⚠️  Guidance:[/dim]")
    
    guidance_map = {
        'fx': {
            'HIGH': "→ FX volatility elevated. Consider hedging currency exposure.",
            'CRITICAL': "→ FX STRESS CRITICAL. Reduce unhedged international positions immediately.",
        },
        'futures': {
            'HIGH': "→ Equity futures stressed. Tighten stops on equity positions.",
            'CRITICAL': "→ EQUITY STRESS CRITICAL. Reduce equity exposure by 50%. Avoid new longs.",
        },
        'rates': {
            'HIGH': "→ Duration risk elevated. Review bond positions for rate sensitivity.",
            'CRITICAL': "→ RATES STRESS CRITICAL. Exit long-duration bonds. Cash is safer.",
        },
        'commodities': {
            'HIGH': "→ Energy/Commodity stress. Review exposure to energy-sensitive sectors.",
            'CRITICAL': "→ COMMODITY STRESS CRITICAL. Avoid energy longs. Consider commodity hedges.",
        },
        'metals': {
            'HIGH': "→ Metals volatility elevated. Gold may signal flight-to-safety.",
            'CRITICAL': "→ METALS STRESS CRITICAL. Consider gold allocation for tail protection.",
        },
    }
    
    for cat_key, stress, level in high_stress:
        if cat_key in guidance_map and level in guidance_map[cat_key]:
            style = "bold red" if level == "CRITICAL" else "yellow"
            console.print(f"     [{style}]{guidance_map[cat_key][level]}[/{style}]")


def _render_metals_guidance(console: Console, metals_result) -> None:
    """Render actionable guidance for Metals risk indicators."""
    temp = metals_result.temperature
    crash_risk = getattr(metals_result, 'crash_risk_pct', 0.0)
    vol_inversions = getattr(metals_result, 'vol_inversion_count', 0)
    
    console.print()
    console.print("  [dim]📊 Guidance:[/dim]")
    
    warnings_issued = False
    
    if vol_inversions >= 2:
        console.print("     [bold red]→ VOLATILITY INVERSION detected in multiple metals.[/bold red]")
        console.print("       [dim]Short-term vol exceeds long-term = stress building. Reduce overnight exposure.[/dim]")
        warnings_issued = True
    
    if crash_risk >= 0.30:
        console.print(f"     [bold red]→ CRASH RISK ELEVATED ({crash_risk:.0%}). Position for downside protection.[/bold red]")
        console.print("       [dim]Consider: tighter stops, reduced size, or put options.[/dim]")
        warnings_issued = True
    elif crash_risk >= 0.15:
        console.print(f"     [yellow]→ Crash risk moderate ({crash_risk:.0%}). Monitor closely for deterioration.[/yellow]")
        warnings_issued = True
    
    metals = getattr(metals_result, 'metals', {})
    for name, metal in metals.items():
        if not getattr(metal, 'data_available', False):
            continue
        ret_5d = getattr(metal, 'return_5d', 0.0)
        if abs(ret_5d) > 0.10:
            direction = "surged" if ret_5d > 0 else "plunged"
            console.print(f"     [yellow]→ {name} {direction} {ret_5d:+.1%} in 5 days. Unusual move - investigate catalyst.[/yellow]")
            warnings_issued = True
    
    if temp >= 1.5:
        console.print("     [bold red]→ EXTREME REGIME: Exit discretionary metals positions. Systematic only.[/bold red]")
        warnings_issued = True
    elif temp >= 1.0:
        console.print("     [yellow]→ Stressed regime: Reduce metals position sizes by 50%.[/yellow]")
        warnings_issued = True
    
    if not warnings_issued:
        console.print("     [bright_green]→ Metals regime normal. Standard position sizing permitted.[/bright_green]")


def _render_equity_guidance(console: Console, market_result) -> None:
    """Render actionable guidance for Equity Market indicators."""
    crash_risk = getattr(market_result, 'crash_risk_pct', 0.0)
    breadth = getattr(market_result, 'breadth', None)
    correlation = getattr(market_result, 'correlation', None)
    
    console.print()
    console.print("  [dim]📊 Guidance:[/dim]")
    
    warnings_issued = False
    
    if breadth:
        pct_above_50ma = getattr(breadth, 'pct_above_50ma', 0.5)
        if pct_above_50ma < 0.25:
            console.print(f"     [bold red]→ BREADTH CRITICAL: Only {pct_above_50ma:.0%} of stocks above 50-day MA.[/bold red]")
            console.print("       [dim]Market held up by few leaders. Distribution phase likely. Raise cash.[/dim]")
            warnings_issued = True
        elif pct_above_50ma < 0.40:
            console.print(f"     [yellow]→ Breadth narrowing: {pct_above_50ma:.0%} above 50-day MA. Watch for breakdown.[/yellow]")
            warnings_issued = True
    
    if correlation:
        avg_corr = getattr(correlation, 'avg_correlation', 0.0)
        if avg_corr > 0.75:
            console.print(f"     [bold red]→ CORRELATION SPIKE: Avg correlation {avg_corr:.0%} = systemic risk.[/bold red]")
            console.print("       [dim]Diversification failing. All assets moving together. Reduce gross exposure.[/dim]")
            warnings_issued = True
        elif avg_corr > 0.60:
            console.print(f"     [yellow]→ Correlations elevated ({avg_corr:.0%}). Diversification benefits reduced.[/yellow]")
            warnings_issued = True
    
    if getattr(market_result, 'exit_signal', False):
        reason = getattr(market_result, 'exit_reason', 'Multiple factors')
        console.print(f"     [bold red]→ EXIT SIGNAL ACTIVE: {reason}[/bold red]")
        console.print("       [dim]Consider moving to cash/bonds until conditions normalize.[/dim]")
        warnings_issued = True
    
    if crash_risk >= 0.30:
        console.print(f"     [bold red]→ Equity crash risk elevated ({crash_risk:.0%}). Hedge tail risk.[/bold red]")
        warnings_issued = True
    
    scale = market_result.scale_factor
    if scale < 0.50:
        console.print(f"     [yellow]→ Position scale reduced to {scale:.0%}. Size all new positions at {scale:.0%} of normal.[/yellow]")
        warnings_issued = True
    
    if not warnings_issued:
        console.print("     [bright_green]→ Market conditions normal. Full position sizing permitted.[/bright_green]")


# =============================================================================
# COMPUTE AND RENDER UNIFIED RISK DASHBOARD
# =============================================================================

def compute_and_render_unified_risk(
    start_date: str = "2020-01-01",
    suppress_output: bool = True,
    console: Console = None,
    output_json: bool = False,
    use_parallel: bool = True,
) -> Dict[str, Any]:
    """
    Compute and render unified risk dashboard from all temperature modules.
    
    This function directly calls the render functions from each module,
    ensuring all data is displayed in a unified view.
    
    February 2026: Added parallel computation using maximum processors for speed.
    
    Args:
        start_date: Start date for historical data
        suppress_output: Whether to suppress stdout/stderr during computation
        console: Rich console for output
        output_json: Return JSON dict instead of rendering
        use_parallel: Use parallel processing for speed (default: True)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing
    
    from decision.risk_temperature import (
        compute_risk_temperature,
        render_risk_temperature_summary,
        RiskTemperatureResult,
    )
    from decision.metals_risk_temperature import (
        compute_anticipatory_metals_risk_temperature,
        render_metals_risk_temperature,
        MetalsRiskTemperatureResult,
        clear_metals_risk_temperature_cache,
    )
    from decision.market_temperature import (
        compute_market_temperature,
        render_market_temperature,
        MarketTemperatureResult,
    )
    
    if console is None:
        console = Console()
    
    # Use maximum available processors
    max_workers = min(3, multiprocessing.cpu_count())  # 3 modules max
    
    # Clear caches to ensure fresh data (avoid stale/scrambled data from previous runs)
    clear_metals_risk_temperature_cache()
    
    # Define computation functions
    def compute_risk():
        return compute_risk_temperature(start_date=start_date)
    
    def compute_metals():
        return compute_anticipatory_metals_risk_temperature(start_date=start_date)
    
    def compute_market():
        return compute_market_temperature(start_date=start_date)
    
    # Compute all three temperature modules in parallel
    # NOTE: We don't use _SuppressOutput with parallel execution because
    # FD-level stdout/stderr redirection is not thread-safe and causes
    # data corruption in yfinance downloads (February 2026 fix)
    if use_parallel and max_workers > 1:
        # Parallel execution - no output suppression to avoid thread safety issues
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_risk = executor.submit(compute_risk)
            future_metals = executor.submit(compute_metals)
            future_market = executor.submit(compute_market)
            
            risk_result = future_risk.result()
            metals_result, alerts, quality_report = future_metals.result()
            market_result = future_market.result()
    else:
        # Sequential fallback
        if suppress_output:
            with _SuppressOutput():
                risk_result = compute_risk_temperature(start_date=start_date)
                metals_result, alerts, quality_report = compute_anticipatory_metals_risk_temperature(start_date=start_date)
                market_result = compute_market_temperature(start_date=start_date)
        else:
            risk_result = compute_risk_temperature(start_date=start_date)
            metals_result, alerts, quality_report = compute_anticipatory_metals_risk_temperature(start_date=start_date)
            market_result = compute_market_temperature(start_date=start_date)
    
    # If JSON output requested, return combined dict
    if output_json:
        return {
            "risk_temperature": risk_result.to_dict(),
            "metals_risk_temperature": metals_result.to_dict(),
            "market_temperature": market_result.to_dict(),
            "computed_at": datetime.now().isoformat(),
        }
    
    # Calculate combined metrics for the header
    combined_temp = (
        0.4 * risk_result.temperature +
        0.3 * metals_result.temperature +
        0.3 * market_result.temperature
    )
    combined_scale = min(
        risk_result.scale_factor,
        metals_result.scale_factor,
        market_result.scale_factor,
    )
    
    if combined_temp < 0.3:
        combined_status = "Calm"
        action = "Full exposure permitted"
    elif combined_temp < 0.7:
        combined_status = "Elevated"
        action = "Monitor positions closely"
    elif combined_temp < 1.2:
        combined_status = "Stressed"
        action = "Reduce risk exposure"
    else:
        combined_status = "Crisis"
        action = "Capital preservation mode"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RENDER COMBINED HEADER
    # ═══════════════════════════════════════════════════════════════════════════
    
    console.print()
    console.print()
    
    title_content = Text()
    title_content.append("\n", style="")
    title_content.append("U N I F I E D   R I S K   D A S H B O A R D", style="bold bright_white")
    title_content.append("\n", style="")
    title_content.append("Cross-Asset • Metals • Equity Market", style="dim")
    title_content.append("\n", style="")
    
    title_panel = Panel(
        Align.center(title_content),
        box=box.DOUBLE,
        border_style="bright_cyan",
        padding=(0, 4),
    )
    console.print(Align.center(title_panel, width=60))
    console.print()
    
    # Combined temperature hero
    temp = combined_temp
    if temp < 0.3:
        status_color = "bright_green"
    elif temp < 0.7:
        status_color = "yellow"
    elif temp < 1.2:
        status_color = "bright_red"
    else:
        status_color = "bold red"
    
    console.print("  [dim]Combined Risk Temperature[/dim]")
    console.print()
    
    hero = Text()
    hero.append("  ")
    hero.append(f"{temp:.2f}", style=f"bold {status_color}")
    hero.append("  ")
    hero.append(combined_status, style=status_color)
    hero.append("  │  Scale: ", style="dim")
    hero.append(f"{combined_scale:.0%}", style=status_color)
    console.print(hero)
    console.print()
    
    # Combined gauge
    gauge = Text()
    gauge.append("  ")
    gauge_width = 50
    filled = int(min(1.0, temp / 2.0) * gauge_width)
    
    for i in range(gauge_width):
        if i < filled:
            segment_pct = i / gauge_width
            if segment_pct < 0.25:
                gauge.append("━", style="bright_green")
            elif segment_pct < 0.5:
                gauge.append("━", style="yellow")
            elif segment_pct < 0.75:
                gauge.append("━", style="bright_red")
            else:
                gauge.append("━", style="bold red")
        else:
            gauge.append("━", style="bright_black")
    
    console.print(gauge)
    
    labels = Text()
    labels.append("  ")
    labels.append("0", style="dim")
    labels.append(" " * 23)
    labels.append("1", style="dim")
    labels.append(" " * 23)
    labels.append("2", style="dim")
    console.print(labels)
    console.print()
    
    console.print(f"  [dim italic]{action}[/dim italic]")
    console.print()
    
    # Quick summary table
    summary_table = Table(
        show_header=True,
        header_style="bold white",
        border_style="dim",
        box=box.SIMPLE,
        padding=(0, 1),
    )
    summary_table.add_column("Module", justify="left", width=14)
    summary_table.add_column("Temp", justify="center", width=6)
    summary_table.add_column("Scale", justify="center", width=7)
    summary_table.add_column("Status", justify="left", width=10)
    summary_table.add_column("Crash", justify="right", width=6)
    summary_table.add_column("Level", justify="left", width=10)
    
    def temp_style(t: float) -> str:
        if t < 0.3: return "bright_green"
        elif t < 0.7: return "yellow"
        elif t < 1.2: return "bright_red"
        return "bold red"
    
    def crash_style(level: str) -> str:
        if level == "Extreme": return "bold red"
        elif level == "High": return "red"
        elif level == "Elevated": return "yellow"
        return "green"
    
    # Risk Temperature row
    r_crash = getattr(risk_result, 'crash_risk_pct', 0.0)
    r_crash_level = getattr(risk_result, 'crash_risk_level', 'Low')
    r_status = getattr(risk_result, 'status', 'Normal') or "Normal"
    summary_table.add_row(
        "Cross-Asset",
        Text(f"{risk_result.temperature:.2f}", style=temp_style(risk_result.temperature)),
        f"{risk_result.scale_factor:.0%}",
        Text(r_status, style=temp_style(risk_result.temperature)),
        Text(f"{r_crash:.0%}", style=crash_style(r_crash_level)),
        Text(r_crash_level, style=crash_style(r_crash_level)),
    )
    
    # Metals Temperature row
    m_crash_level = metals_result.crash_risk_level
    summary_table.add_row(
        "Metals",
        Text(f"{metals_result.temperature:.2f}", style=temp_style(metals_result.temperature)),
        f"{metals_result.scale_factor:.0%}",
        Text(metals_result.regime_state, style=temp_style(metals_result.temperature)),
        Text(f"{metals_result.crash_risk_pct:.0%}", style=crash_style(m_crash_level)),
        Text(m_crash_level, style=crash_style(m_crash_level)),
    )
    
    # Market Temperature row
    mk_crash_level = market_result.crash_risk_level
    summary_table.add_row(
        "Equity Market",
        Text(f"{market_result.temperature:.2f}", style=temp_style(market_result.temperature)),
        f"{market_result.scale_factor:.0%}",
        Text(market_result.status, style=temp_style(market_result.temperature)),
        Text(f"{market_result.crash_risk_pct:.0%}", style=crash_style(mk_crash_level)),
        Text(mk_crash_level, style=crash_style(mk_crash_level)),
    )
    
    console.print(summary_table)
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: CROSS-ASSET RISK TEMPERATURE
    # ═══════════════════════════════════════════════════════════════════════════
    
    console.print(Rule(" Cross-Asset Stress ", style="bright_cyan"))
    console.print()
    
    # Render stress categories
    categories = risk_result.categories
    if categories:
        category_config = [
            ("fx", "FX Carry"),
            ("futures", "Equities"),
            ("rates", "Duration"),
            ("commodities", "Commodities"),  # Includes Oil, Copper, Gold, Silver
            ("metals", "Metals"),
        ]
        
        for cat_key, cat_label in category_config:
            if cat_key not in categories:
                continue
            
            cat = categories[cat_key]
            stress = getattr(cat, 'stress_level', 0.0)
            
            if stress < 0.5:
                stress_style = "bright_green"
            elif stress < 1.0:
                stress_style = "yellow"
            elif stress < 1.5:
                stress_style = "bright_red"
            else:
                stress_style = "bold red"
            
            # Compute crash risk score (0-100) from stress level
            crash_risk_score = int(min(100, (stress / 2.0) * 100))
            if crash_risk_score >= 70:
                risk_style = "bold red"
            elif crash_risk_score >= 50:
                risk_style = "red"
            elif crash_risk_score >= 30:
                risk_style = "yellow"
            else:
                risk_style = "green"
            
            line = Text()
            line.append("  ")
            line.append(f"{cat_label:<12}", style="dim")
            
            mini_width = 20
            mini_filled = int(min(1.0, stress / 2.0) * mini_width)
            for i in range(mini_width):
                if i < mini_filled:
                    line.append("━", style=stress_style)
                else:
                    line.append("━", style="bright_black")
            
            line.append(f"  {stress:.2f}", style=stress_style)
            
            # Add crash risk score
            line.append(f"  Risk: ", style="dim")
            line.append(f"{crash_risk_score}", style=risk_style)
            
            # Show top indicator
            indicators = getattr(cat, 'indicators', [])
            if indicators:
                top_ind = max(indicators, key=lambda x: abs(getattr(x, 'zscore', 0.0)))
                if getattr(top_ind, 'data_available', False):
                    zscore = getattr(top_ind, 'zscore', 0.0)
                    line.append(f"  ({top_ind.name}: z={zscore:+.1f})", style="dim")
            
            console.print(line)
    
    # Overnight budget
    if risk_result.overnight_budget_active:
        console.print()
        overnight_line = Text()
        overnight_line.append("  ")
        overnight_line.append("Overnight Cap   ", style="dim")
        overnight_line.append(f"{risk_result.overnight_max_position:.0%}", style="bold yellow")
        overnight_line.append("  Active", style="dim italic")
        console.print(overnight_line)
    
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: METALS RISK TEMPERATURE
    # ═══════════════════════════════════════════════════════════════════════════
    
    console.print(Rule(" Metals Risk ", style="bright_cyan"))
    console.print()
    
    # Metals ratio indicators
    console.print("  [dim]Ratio Indicators[/dim]")
    console.print()
    
    for ind in metals_result.indicators:
        if not ind.data_available:
            continue
        
        zscore = ind.zscore
        if abs(zscore) < 0.5:
            ind_style = "bright_green"
        elif abs(zscore) < 1.5:
            ind_style = "yellow"
        elif abs(zscore) < 2.5:
            ind_style = "bright_red"
        else:
            ind_style = "bold red"
        
        line = Text()
        line.append("  ")
        line.append(f"{ind.name:<22}", style="dim")
        line.append(f"{ind.value:.3f}".ljust(10), style="white")
        line.append(f"z={zscore:+.2f}".ljust(10), style=ind_style)
        
        interp = getattr(ind, 'interpretation', '')
        if interp:
            line.append(interp, style="dim italic")
        
        console.print(line)
    
    console.print()
    
    # Individual metals table
    console.print("  [dim]Individual Metals[/dim]")
    console.print()
    
    metals_table = Table(
        show_header=True,
        header_style="bold white",
        border_style="dim",
        box=box.SIMPLE,
        padding=(0, 1),
    )
    metals_table.add_column("Metal", justify="left", width=10)
    metals_table.add_column("Price", justify="right", width=10)
    metals_table.add_column("1D", justify="right", width=8)
    metals_table.add_column("5D", justify="right", width=8)
    metals_table.add_column("21D", justify="right", width=8)
    metals_table.add_column("Vol", justify="right", width=6)
    metals_table.add_column("Momentum", justify="left", width=12)
    metals_table.add_column("Risk", justify="right", width=6)
    
    for name, metal in metals_result.metals.items():
        if not metal.data_available:
            continue
        
        # Format price
        if metal.price >= 1000:
            price_str = f"${metal.price:,.0f}"
        else:
            price_str = f"${metal.price:.2f}"
        
        # Returns
        ret_1d_style = "bright_green" if metal.return_1d >= 0 else "indian_red1"
        ret_5d_style = "bright_green" if metal.return_5d >= 0 else "indian_red1"
        ret_21d_style = "bright_green" if metal.return_21d >= 0 else "indian_red1"
        
        # Momentum signal styling
        momentum = getattr(metal, 'momentum_signal', '→ Flat')
        if "Strong" in momentum and "↑" in momentum:
            mom_style = "bold bright_green"
        elif "Rising" in momentum or "↗" in momentum:
            mom_style = "bright_green"
        elif "Weak" in momentum or "↓" in momentum:
            mom_style = "bold indian_red1"
        elif "Falling" in momentum or "↘" in momentum:
            mom_style = "indian_red1"
        else:
            mom_style = "dim"
        
        # Compute individual crash risk score (0-100)
        stress = getattr(metal, 'stress_level', 0.0)
        vol = metal.volatility
        ret_5d = metal.return_5d
        
        # Components: stress (0-40), volatility (0-30), drawdown (0-30)
        stress_pts = (stress / 2.0) * 40
        vol_pts = min(vol / 1.5, 1.0) * 30
        drawdown_pts = min(max(0, -ret_5d) / 0.30, 1.0) * 30
        metal_crash_risk = int(min(100, stress_pts + vol_pts + drawdown_pts))
        
        if metal_crash_risk >= 70:
            metal_risk_style = "bold red"
        elif metal_crash_risk >= 50:
            metal_risk_style = "red"
        elif metal_crash_risk >= 30:
            metal_risk_style = "yellow"
        else:
            metal_risk_style = "green"
        
        metals_table.add_row(
            metal.name,
            price_str,
            Text(f"{metal.return_1d:+.1%}", style=ret_1d_style),
            Text(f"{metal.return_5d:+.1%}", style=ret_5d_style),
            Text(f"{metal.return_21d:+.1%}", style=ret_21d_style),
            f"{metal.volatility:.0%}",
            Text(momentum, style=mom_style),
            Text(f"{metal_crash_risk}", style=metal_risk_style),
        )
    
    console.print(metals_table)
    console.print()
    
    # Metals governance info
    gov_line = Text()
    gov_line.append("  ")
    gov_line.append("Regime: ", style="dim")
    gov_line.append(metals_result.regime_state, style=temp_style(metals_result.temperature))
    gov_line.append("   Overnight Exposure: ", style="dim")
    gov_line.append(f"{metals_result.gap_risk_estimate:.1%}", style="white")
    if metals_result.crash_risk_pct > 0.05:
        gov_line.append("   Crash Risk: ", style="dim")
        crash_style = "bold red" if metals_result.crash_risk_level == "Extreme" else ("red" if metals_result.crash_risk_level == "High" else "yellow")
        gov_line.append(f"{metals_result.crash_risk_pct:.0%}", style=crash_style)
    console.print(gov_line)
    
    # Add metals actionable guidance
    _render_metals_guidance(console, metals_result)
    
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: EQUITY MARKET TEMPERATURE
    # ═══════════════════════════════════════════════════════════════════════════
    
    console.print(Rule(" Equity Market ", style="bright_cyan"))
    console.print()
    
    # Universe metrics table
    console.print("  [dim]Universe Metrics[/dim]")
    console.print()
    
    univ_table = Table(
        show_header=True,
        header_style="bold white",
        border_style="dim",
        box=box.SIMPLE,
        padding=(0, 1),
    )
    univ_table.add_column("Universe", justify="left", width=14)
    univ_table.add_column("1D", justify="right", width=8)
    univ_table.add_column("5D", justify="right", width=8)
    univ_table.add_column("21D", justify="right", width=8)
    univ_table.add_column("Vol", justify="right", width=6)
    univ_table.add_column("Momentum", justify="left", width=12)
    univ_table.add_column("Risk", justify="right", width=6)
    
    for name, univ in market_result.universes.items():
        if not univ.data_available:
            continue
        
        ret_1d_style = "bright_green" if univ.return_1d >= 0 else "indian_red1"
        ret_5d_style = "bright_green" if univ.return_5d >= 0 else "indian_red1"
        ret_21d_style = "bright_green" if univ.return_21d >= 0 else "indian_red1"
        
        vol_style = "red" if univ.volatility_percentile > 0.8 else ("yellow" if univ.volatility_percentile > 0.6 else "white")
        
        momentum = univ.momentum_signal
        if "Strong" in momentum and "↑" in momentum:
            mom_style = "bold bright_green"
        elif "Rising" in momentum:
            mom_style = "bright_green"
        elif "Weak" in momentum or "↓" in momentum:
            mom_style = "bold indian_red1"
        elif "Falling" in momentum:
            mom_style = "indian_red1"
        else:
            mom_style = "dim"
        
        # Compute individual crash risk score (0-100)
        stress = getattr(univ, 'stress_level', 0.0)
        vol_pctl = univ.volatility_percentile
        vol_inverted = univ.vol_inverted
        ret_5d = univ.return_5d
        
        # Components: stress (0-30), vol_pctl (0-25), vol_inversion (0-20), drawdown (0-25)
        stress_pts = (stress / 2.0) * 30
        vol_pts = vol_pctl * 25
        inversion_pts = 20 if vol_inverted else 0
        drawdown_pts = min(max(0, -ret_5d) / 0.15, 1.0) * 25
        univ_crash_risk = int(min(100, stress_pts + vol_pts + inversion_pts + drawdown_pts))
        
        if univ_crash_risk >= 70:
            univ_risk_style = "bold red"
        elif univ_crash_risk >= 50:
            univ_risk_style = "red"
        elif univ_crash_risk >= 30:
            univ_risk_style = "yellow"
        else:
            univ_risk_style = "green"
        
        univ_table.add_row(
            univ.name,
            Text(f"{univ.return_1d:+.1%}", style=ret_1d_style),
            Text(f"{univ.return_5d:+.1%}", style=ret_5d_style),
            Text(f"{univ.return_21d:+.1%}", style=ret_21d_style),
            Text(f"{univ.volatility_20d:.0%}", style=vol_style),
            Text(momentum, style=mom_style),
            Text(f"{univ_crash_risk}", style=univ_risk_style),
        )
    
    console.print(univ_table)
    console.print()
    
    # Sector-by-Sector Breakdown (February 2026)
    sectors = getattr(market_result, 'sectors', {})
    
    # Debug: Show sector count
    available_sectors = [s for s in sectors.values() if s.data_available] if sectors else []
    
    if available_sectors:
        console.print("  [dim]Sector Breakdown[/dim]")
        console.print()
        
        sector_table = Table(
            show_header=True,
            header_style="bold white",
            border_style="dim",
            box=box.SIMPLE,
            padding=(0, 1),
        )
        sector_table.add_column("Sector", justify="left", width=14)
        sector_table.add_column("1D", justify="right", width=8)
        sector_table.add_column("5D", justify="right", width=8)
        sector_table.add_column("21D", justify="right", width=8)
        sector_table.add_column("Vol", justify="right", width=6)
        sector_table.add_column("Momentum", justify="left", width=12)
        sector_table.add_column("Risk", justify="right", width=6)
        
        # Sort sectors by risk score (highest first)
        sorted_sectors = sorted(
            sectors.values(),
            key=lambda s: s.risk_score if s.data_available else -1,
            reverse=True
        )
        
        for sector in sorted_sectors:
            if not sector.data_available:
                continue
            
            ret_1d_style = "bright_green" if sector.return_1d >= 0 else "indian_red1"
            ret_5d_style = "bright_green" if sector.return_5d >= 0 else "indian_red1"
            ret_21d_style = "bright_green" if sector.return_21d >= 0 else "indian_red1"
            
            vol_style = "red" if sector.volatility_percentile > 0.8 else ("yellow" if sector.volatility_percentile > 0.6 else "white")
            
            momentum = sector.momentum_signal
            if "Strong" in momentum and "↑" in momentum:
                mom_style = "bold bright_green"
            elif "Rising" in momentum or "↗" in momentum:
                mom_style = "bright_green"
            elif "Weak" in momentum or "↓" in momentum:
                mom_style = "bold indian_red1"
            elif "Falling" in momentum or "↘" in momentum:
                mom_style = "indian_red1"
            else:
                mom_style = "dim"
            
            risk_score = sector.risk_score
            if risk_score >= 70:
                risk_style = "bold red"
            elif risk_score >= 50:
                risk_style = "red"
            elif risk_score >= 30:
                risk_style = "yellow"
            else:
                risk_style = "green"
            
            sector_table.add_row(
                sector.name,
                Text(f"{sector.return_1d:+.1%}", style=ret_1d_style),
                Text(f"{sector.return_5d:+.1%}", style=ret_5d_style),
                Text(f"{sector.return_21d:+.1%}", style=ret_21d_style),
                Text(f"{sector.volatility_20d:.0%}", style=vol_style),
                Text(momentum, style=mom_style),
                Text(f"{risk_score}", style=risk_style),
            )
        
        console.print(sector_table)
        console.print()
    
    # Currency Pairs Breakdown (February 2026)
    currencies = getattr(market_result, 'currencies', {})
    available_currencies = [c for c in currencies.values() if c.data_available] if currencies else []
    
    if available_currencies:
        console.print("  [dim]Currency Pairs[/dim]")
        console.print()
        
        currency_table = Table(
            show_header=True,
            header_style="bold white",
            border_style="dim",
            box=box.SIMPLE,
            padding=(0, 1),
        )
        currency_table.add_column("Pair", justify="left", width=10)
        currency_table.add_column("Rate", justify="right", width=10)
        currency_table.add_column("1D", justify="right", width=8)
        currency_table.add_column("5D", justify="right", width=8)
        currency_table.add_column("21D", justify="right", width=8)
        currency_table.add_column("Momentum", justify="left", width=12)
        currency_table.add_column("Risk", justify="right", width=6)
        
        # Sort currencies by risk score (highest first)
        sorted_currencies = sorted(
            available_currencies,
            key=lambda c: c.risk_score,
            reverse=True
        )
        
        for currency in sorted_currencies:
            ret_1d_style = "bright_green" if currency.return_1d >= 0 else "indian_red1"
            ret_5d_style = "bright_green" if currency.return_5d >= 0 else "indian_red1"
            ret_21d_style = "bright_green" if currency.return_21d >= 0 else "indian_red1"
            
            momentum = currency.momentum_signal
            if "Strong" in momentum and "↑" in momentum:
                mom_style = "bold bright_green"
            elif "Rising" in momentum or "↗" in momentum:
                mom_style = "bright_green"
            elif "Weak" in momentum or "↓" in momentum:
                mom_style = "bold indian_red1"
            elif "Falling" in momentum or "↘" in momentum:
                mom_style = "indian_red1"
            else:
                mom_style = "dim"
            
            risk_score = currency.risk_score
            if risk_score >= 70:
                risk_style = "bold red"
            elif risk_score >= 50:
                risk_style = "red"
            elif risk_score >= 30:
                risk_style = "yellow"
            else:
                risk_style = "green"
            
            # Format rate based on pair convention
            if "BTC" in currency.name or "ETH" in currency.name:
                # Crypto - show as currency with comma separator
                rate_str = f"${currency.rate:,.0f}"
            elif "JPY" in currency.name:
                rate_str = f"{currency.rate:.2f}"
            else:
                rate_str = f"{currency.rate:.4f}"
            
            currency_table.add_row(
                currency.name,
                rate_str,
                Text(f"{currency.return_1d:+.1%}", style=ret_1d_style),
                Text(f"{currency.return_5d:+.1%}", style=ret_5d_style),
                Text(f"{currency.return_21d:+.1%}", style=ret_21d_style),
                Text(momentum, style=mom_style),
                Text(f"{risk_score}", style=risk_style),
            )
        
        console.print(currency_table)
        console.print()
    
    # Market Breadth
    console.print("  [dim]Market Breadth[/dim]")
    console.print()
    
    breadth = market_result.breadth
    breadth_line = Text()
    breadth_line.append("  ")
    breadth_line.append("Above 50MA: ", style="dim")
    
    pct_50ma = breadth.pct_above_50ma
    if pct_50ma < 0.30:
        b_style = "bold red"
    elif pct_50ma < 0.50:
        b_style = "yellow"
    else:
        b_style = "green"
    breadth_line.append(f"{pct_50ma:.0%}", style=b_style)
    
    breadth_line.append("   Above 200MA: ", style="dim")
    breadth_line.append(f"{breadth.pct_above_200ma:.0%}", style="white")
    
    breadth_line.append("   A/D Ratio: ", style="dim")
    ad_style = "green" if breadth.advance_decline_ratio > 1 else "red"
    breadth_line.append(f"{breadth.advance_decline_ratio:.2f}", style=ad_style)
    console.print(breadth_line)
    
    interp_line = Text()
    interp_line.append("  ")
    interp_line.append(breadth.interpretation, style="dim italic")
    console.print(interp_line)
    console.print()
    
    # Correlation Stress (with copula-based tail dependence)
    console.print("  [dim]Correlation & Tail Dependence[/dim]")
    console.print()
    
    corr = market_result.correlation
    
    # Traditional correlation
    corr_line = Text()
    corr_line.append("  ")
    corr_line.append("Avg Correlation: ", style="dim")
    
    if corr.systemic_risk_elevated:
        c_style = "bold red"
    elif corr.avg_correlation > 0.60:
        c_style = "yellow"
    else:
        c_style = "green"
    corr_line.append(f"{corr.avg_correlation:.0%}", style=c_style)
    
    # Show method used
    method = getattr(corr, 'method', 'pearson')
    corr_line.append(f"  ({method})", style="dim italic")
    console.print(corr_line)
    
    # Copula-based tail dependence (if available)
    lower_tail = getattr(corr, 'lower_tail_dependence_avg', 0.0)
    crash_contagion = getattr(corr, 'crash_contagion_risk', 0.0)
    
    if lower_tail > 0.01 or crash_contagion > 0.01:
        tail_line = Text()
        tail_line.append("  ")
        tail_line.append("Lower Tail Dep: ", style="dim")
        
        if lower_tail >= 0.45:
            t_style = "bold red"
        elif lower_tail >= 0.25:
            t_style = "yellow"
        else:
            t_style = "green"
        tail_line.append(f"{lower_tail:.0%}", style=t_style)
        
        tail_line.append("   Crash Contagion: ", style="dim")
        if crash_contagion >= 0.5:
            cc_style = "bold red"
        elif crash_contagion >= 0.25:
            cc_style = "yellow"
        else:
            cc_style = "green"
        tail_line.append(f"{crash_contagion:.0%}", style=cc_style)
        console.print(tail_line)
    
    # Interpretation
    interp_line2 = Text()
    interp_line2.append("  ")
    interp_line2.append(corr.interpretation, style="dim italic")
    console.print(interp_line2)
    console.print()
    
    # Market momentum summary
    summary_line = Text()
    summary_line.append("  ")
    summary_line.append("Momentum: ", style="dim")
    
    mom = market_result.overall_momentum
    if "Bullish" in mom:
        mom_style = "bright_green"
    elif "Bearish" in mom:
        mom_style = "bright_red"
    else:
        mom_style = "yellow"
    summary_line.append(mom, style=mom_style)
    
    summary_line.append("   Sector Rotation: ", style="dim")
    rot = market_result.sector_rotation_signal
    if "Risk-On" in rot:
        rot_style = "bright_green"
    elif "Risk-Off" in rot or "Defensive" in rot:
        rot_style = "bright_red"
    else:
        rot_style = "yellow"
    summary_line.append(rot, style=rot_style)
    console.print(summary_line)
    
    # Add equity market actionable guidance
    _render_equity_guidance(console, market_result)
    
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: MARKET DIRECTION (Indices + Universes + Sectors)
    # ═══════════════════════════════════════════════════════════════════════════
    
    console.print(Rule(" Market Direction ", style="bright_cyan"))
    console.print()
    
    # Fetch market direction data
    try:
        from ingestion.data_utils import get_market_direction_summary
        direction_summary = get_market_direction_summary(force_refresh=False)
        
        if "error" not in direction_summary:
            # Market Trend Header
            market_trend = direction_summary.get("market_trend", "Unknown")
            if market_trend == "Bullish":
                trend_style = "bold bright_green"
            elif market_trend == "Bearish":
                trend_style = "bold red"
            else:
                trend_style = "yellow"
            
            console.print(f"  [dim]Overall Market Trend:[/dim] [{trend_style}]{market_trend}[/{trend_style}]")
            console.print()
            
            # Indices Table (if available)
            indices_df = direction_summary.get("indices")
            if indices_df is not None and not indices_df.empty:
                console.print("  [dim]Major Indices[/dim]")
                console.print()
                
                indices_table = Table(
                    show_header=True,
                    header_style="bold white",
                    border_style="dim",
                    box=box.SIMPLE,
                    padding=(0, 1),
                )
                indices_table.add_column("Index", justify="left", width=16)
                indices_table.add_column("1D", justify="right", width=8)
                indices_table.add_column("1W", justify="right", width=8)
                indices_table.add_column("1M", justify="right", width=8)
                indices_table.add_column("Vol", justify="right", width=6)
                indices_table.add_column("Trend", justify="left", width=12)
                
                for _, row in indices_df.iterrows():
                    name = str(row.get("name", row["symbol"]))[:16]
                    r1 = row.get("ret_1d")
                    r5 = row.get("ret_5d")
                    r21 = row.get("ret_21d")
                    vol = row.get("volatility")
                    trend = str(row.get("trend", ""))
                    
                    r1_style = "bright_green" if r1 and r1 >= 0 else "indian_red1"
                    r5_style = "bright_green" if r5 and r5 >= 0 else "indian_red1"
                    r21_style = "bright_green" if r21 and r21 >= 0 else "indian_red1"
                    
                    if "Rising" in trend:
                        trend_style = "bright_green"
                    elif "Falling" in trend:
                        trend_style = "indian_red1"
                    else:
                        trend_style = "dim"
                    
                    indices_table.add_row(
                        name,
                        Text(f"{r1:+.1f}%" if r1 is not None else "N/A", style=r1_style),
                        Text(f"{r5:+.1f}%" if r5 is not None else "N/A", style=r5_style),
                        Text(f"{r21:+.1f}%" if r21 is not None else "N/A", style=r21_style),
                        f"{vol:.0f}%" if vol is not None else "N/A",
                        Text(trend, style=trend_style),
                    )
                
                console.print(indices_table)
                console.print()
            
            # Market Universe ETFs Table
            universes_df = direction_summary.get("universes")
            if universes_df is not None and not universes_df.empty:
                console.print("  [dim]Market Universe ETFs[/dim]")
                console.print()
                
                universes_table = Table(
                    show_header=True,
                    header_style="bold white",
                    border_style="dim",
                    box=box.SIMPLE,
                    padding=(0, 1),
                )
                universes_table.add_column("Universe", justify="left", width=16)
                universes_table.add_column("1D", justify="right", width=8)
                universes_table.add_column("1W", justify="right", width=8)
                universes_table.add_column("1M", justify="right", width=8)
                universes_table.add_column("Vol", justify="right", width=6)
                universes_table.add_column("Trend", justify="left", width=12)
                
                for _, row in universes_df.iterrows():
                    name = str(row.get("name", row["symbol"]))[:16]
                    r1 = row.get("ret_1d")
                    r5 = row.get("ret_5d")
                    r21 = row.get("ret_21d")
                    vol = row.get("volatility")
                    trend = str(row.get("trend", ""))
                    
                    r1_style = "bright_green" if r1 and r1 >= 0 else "indian_red1"
                    r5_style = "bright_green" if r5 and r5 >= 0 else "indian_red1"
                    r21_style = "bright_green" if r21 and r21 >= 0 else "indian_red1"
                    
                    if "Rising" in trend:
                        trend_style = "bright_green"
                    elif "Falling" in trend:
                        trend_style = "indian_red1"
                    else:
                        trend_style = "dim"
                    
                    universes_table.add_row(
                        name,
                        Text(f"{r1:+.1f}%" if r1 is not None else "N/A", style=r1_style),
                        Text(f"{r5:+.1f}%" if r5 is not None else "N/A", style=r5_style),
                        Text(f"{r21:+.1f}%" if r21 is not None else "N/A", style=r21_style),
                        f"{vol:.0f}%" if vol is not None else "N/A",
                        Text(trend, style=trend_style),
                    )
                
                console.print(universes_table)
                console.print()
            
            # Sector ETFs Table (sorted by momentum)
            sectors_df = direction_summary.get("sectors")
            if sectors_df is not None and not sectors_df.empty:
                console.print("  [dim]Sector ETFs (sorted by momentum)[/dim]")
                console.print()
                
                sectors_table = Table(
                    show_header=True,
                    header_style="bold white",
                    border_style="dim",
                    box=box.SIMPLE,
                    padding=(0, 1),
                )
                sectors_table.add_column("Sector", justify="left", width=16)
                sectors_table.add_column("1D", justify="right", width=8)
                sectors_table.add_column("1W", justify="right", width=8)
                sectors_table.add_column("1M", justify="right", width=8)
                sectors_table.add_column("Vol", justify="right", width=6)
                sectors_table.add_column("Trend", justify="left", width=12)
                
                for _, row in sectors_df.iterrows():
                    name = str(row.get("name", row["symbol"]))[:16]
                    r1 = row.get("ret_1d")
                    r5 = row.get("ret_5d")
                    r21 = row.get("ret_21d")
                    vol = row.get("volatility")
                    trend = str(row.get("trend", ""))
                    
                    r1_style = "bright_green" if r1 and r1 >= 0 else "indian_red1"
                    r5_style = "bright_green" if r5 and r5 >= 0 else "indian_red1"
                    r21_style = "bright_green" if r21 and r21 >= 0 else "indian_red1"
                    
                    if "Rising" in trend:
                        trend_style = "bright_green"
                    elif "Falling" in trend:
                        trend_style = "indian_red1"
                    else:
                        trend_style = "dim"
                    
                    sectors_table.add_row(
                        name,
                        Text(f"{r1:+.1f}%" if r1 is not None else "N/A", style=r1_style),
                        Text(f"{r5:+.1f}%" if r5 is not None else "N/A", style=r5_style),
                        Text(f"{r21:+.1f}%" if r21 is not None else "N/A", style=r21_style),
                        f"{vol:.0f}%" if vol is not None else "N/A",
                        Text(trend, style=trend_style),
                    )
                
                console.print(sectors_table)
                console.print()
            
            # Leaders and Laggards
            leaders = direction_summary.get("leaders", [])
            laggards = direction_summary.get("laggards", [])
            
            if leaders or laggards:
                ll_line = Text()
                ll_line.append("  ")
                
                if leaders:
                    ll_line.append("Leaders: ", style="dim")
                    for i, leader in enumerate(leaders[:3]):
                        if i > 0:
                            ll_line.append(", ", style="dim")
                        ret = leader.get('ret_21d', 0) or 0
                        ll_line.append(f"{leader['name']} ({ret:+.1f}%)", style="bright_green")
                
                if laggards:
                    ll_line.append("   Laggards: ", style="dim")
                    for i, laggard in enumerate(laggards[:3]):
                        if i > 0:
                            ll_line.append(", ", style="dim")
                        ret = laggard.get('ret_21d', 0) or 0
                        ll_line.append(f"{laggard['name']} ({ret:+.1f}%)", style="indian_red1")
                
                console.print(ll_line)
                console.print()
        else:
            console.print(f"  [dim]Market direction data unavailable: {direction_summary.get('error', 'Unknown error')}[/dim]")
            console.print()
    except Exception as e:
        console.print(f"  [dim]Market direction unavailable: {str(e)}[/dim]")
        console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FOOTER: DATA QUALITY & TIMESTAMP
    # ═══════════════════════════════════════════════════════════════════════════
    
    console.print(Rule(style="dim"))
    console.print()
    
    # Combined data quality
    avg_quality = (
        risk_result.data_quality +
        metals_result.data_quality +
        market_result.data_quality
    ) / 3.0
    
    quality_style = "green" if avg_quality >= 0.9 else ("yellow" if avg_quality >= 0.7 else "red")
    
    footer_line = Text()
    footer_line.append("  ")
    footer_line.append("Data Quality: ", style="dim")
    footer_line.append(f"{avg_quality:.0%}", style=quality_style)
    footer_line.append("   Computed: ", style="dim")
    footer_line.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), style="dim italic")
    console.print(footer_line)
    console.print()
    console.print()
    
    return {
        "risk_temperature": risk_result.to_dict(),
        "metals_risk_temperature": metals_result.to_dict(),
        "market_temperature": market_result.to_dict(),
        "combined_temperature": combined_temp,
        "combined_scale_factor": combined_scale,
        "combined_status": combined_status,
        "computed_at": datetime.now().isoformat(),
    }


# =============================================================================
# STANDALONE CLI
# =============================================================================

if __name__ == "__main__":
    """Run unified risk dashboard."""
    parser = argparse.ArgumentParser(
        description="Unified Risk Dashboard — Combined Risk Temperature View"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--start", default="2020-01-01", help="Start date for historical data")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output during computation")
    args = parser.parse_args()
    
    # Suppress output during computation
    suppress = not args.verbose
    
    console = Console()
    
    if not args.json:
        console.print()
        console.print("  [dim]Computing unified risk assessment...[/dim]")
    
    # Compute and render
    result = compute_and_render_unified_risk(
        start_date=args.start,
        suppress_output=suppress,
        console=console if not args.json else None,
        output_json=args.json,
    )
    
    # Wait for any lingering threads
    time.sleep(0.5)
    sys.stdout.flush()
    sys.stderr.flush()
    
    if args.json:
        # Clear any stray output
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()
        print(json.dumps(result, indent=2))
