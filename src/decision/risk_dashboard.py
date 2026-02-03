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
# COMPUTE AND RENDER UNIFIED RISK DASHBOARD
# =============================================================================

def compute_and_render_unified_risk(
    start_date: str = "2020-01-01",
    suppress_output: bool = True,
    console: Console = None,
    output_json: bool = False,
) -> Dict[str, Any]:
    """
    Compute and render unified risk dashboard from all temperature modules.
    
    This function directly calls the render functions from each module,
    ensuring all data is displayed in a unified view.
    """
    from decision.risk_temperature import (
        compute_risk_temperature,
        render_risk_temperature_summary,
        RiskTemperatureResult,
    )
    from decision.metals_risk_temperature import (
        compute_anticipatory_metals_risk_temperature,
        render_metals_risk_temperature,
        MetalsRiskTemperatureResult,
    )
    from decision.market_temperature import (
        compute_market_temperature,
        render_market_temperature,
        MarketTemperatureResult,
    )
    
    if console is None:
        console = Console()
    
    # Compute all three temperature modules
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
    summary_table.add_column("Module", justify="left", width=15)
    summary_table.add_column("Temp", justify="center", width=8)
    summary_table.add_column("Scale", justify="center", width=8)
    summary_table.add_column("Status", justify="left", width=12)
    summary_table.add_column("Crash Risk", justify="center", width=12)
    
    def temp_style(t: float) -> str:
        if t < 0.3: return "bright_green"
        elif t < 0.7: return "yellow"
        elif t < 1.2: return "bright_red"
        return "bold red"
    
    # Risk Temperature row
    r_crash = getattr(risk_result, 'crash_risk_pct', 0.0)
    r_crash_level = getattr(risk_result, 'crash_risk_level', 'Low')
    summary_table.add_row(
        "Cross-Asset",
        Text(f"{risk_result.temperature:.2f}", style=temp_style(risk_result.temperature)),
        f"{risk_result.scale_factor:.0%}",
        getattr(risk_result, 'status', 'Normal') or "Normal",
        f"{r_crash:.0%} ({r_crash_level})" if r_crash > 0.01 else "—",
    )
    
    # Metals Temperature row
    summary_table.add_row(
        "Metals",
        Text(f"{metals_result.temperature:.2f}", style=temp_style(metals_result.temperature)),
        f"{metals_result.scale_factor:.0%}",
        metals_result.regime_state,
        f"{metals_result.crash_risk_pct:.0%} ({metals_result.crash_risk_level})",
    )
    
    # Market Temperature row
    summary_table.add_row(
        "Equity Market",
        Text(f"{market_result.temperature:.2f}", style=temp_style(market_result.temperature)),
        f"{market_result.scale_factor:.0%}",
        market_result.status,
        f"{market_result.crash_risk_pct:.0%} ({market_result.crash_risk_level})",
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
            ("commodities", "Energy"),
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
    metals_table.add_column("Vol", justify="right", width=8)
    metals_table.add_column("Momentum", justify="left", width=16)
    
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
        
        metals_table.add_row(
            metal.name,
            price_str,
            Text(f"{metal.return_1d:+.1%}", style=ret_1d_style),
            Text(f"{metal.return_5d:+.1%}", style=ret_5d_style),
            Text(f"{metal.return_21d:+.1%}", style=ret_21d_style),
            f"{metal.volatility:.0%}",
            Text(momentum, style=mom_style),
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
    univ_table.add_column("Vol", justify="right", width=8)
    univ_table.add_column("Momentum", justify="left", width=18)
    
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
        
        univ_table.add_row(
            univ.name,
            Text(f"{univ.return_1d:+.1%}", style=ret_1d_style),
            Text(f"{univ.return_5d:+.1%}", style=ret_5d_style),
            Text(f"{univ.return_21d:+.1%}", style=ret_21d_style),
            Text(f"{univ.volatility_20d:.0%}", style=vol_style),
            Text(momentum, style=mom_style),
        )
    
    console.print(univ_table)
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
    
    # Correlation Stress
    console.print("  [dim]Correlation Stress[/dim]")
    console.print()
    
    corr = market_result.correlation
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
    corr_line.append("   ", style="")
    corr_line.append(corr.interpretation, style="dim italic")
    console.print(corr_line)
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
