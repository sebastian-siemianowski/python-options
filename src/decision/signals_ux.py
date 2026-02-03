#!/usr/bin/env python3
"""
signals_ux.py

Presentation layer for FX signals - handles all output formatting and display logic.
Separates presentation concerns from core signal computation logic for better modularity.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.live import Live
from rich.layout import Layout
from rich.spinner import Spinner
from rich.padding import Padding
from rich.rule import Rule
from rich.align import Align
from contextlib import contextmanager


def convert_to_float(x) -> float:
    """Convert various numeric types to float, handling edge cases gracefully."""
    try:
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        if hasattr(x, "item"):
            return float(x.item())
        arr = np.asarray(x)
        if arr.size == 1:
            return float(arr.reshape(()).item())
        return float("nan")
    except Exception:
        return float("nan")


def format_horizon_label(horizon_days: int) -> str:
    """Convert trading days to human-readable timeframe labels."""
    mapping = {
        1: "1 day",
        2: "2 days",
        3: "3 days",
        5: "1 week (5d)",
        7: "1 week",
        10: "2 weeks (10d)",
        14: "2 weeks",
        21: "1 month",
        42: "2 months",
        63: "3 months",
        84: "4 months",
        105: "5 months",
        126: "6 months",
        189: "9 months",
        252: "12 months"
    }
    return mapping.get(horizon_days, f"{horizon_days} days")


def build_asset_display_label(asset_symbol: str, full_title: str) -> str:
    """Build a concise display like 'Ticker (Name)' when available.
    
    Args:
        asset_symbol: The trading symbol/ticker
        full_title: Full descriptive title which may contain symbol in parentheses
        
    Returns:
        Formatted display label combining symbol and name
    """
    try:
        name_part = full_title.split(" — ")[0].strip()
    except Exception:
        name_part = full_title.strip()
    
    # If name_part already contains parentheses (e.g., "Company Name (TICKER)"), use it
    if "(" in name_part and ")" in name_part:
        return name_part
    
    # Otherwise prepend the asset symbol for clarity
    if asset_symbol:
        return f"{asset_symbol} — {name_part}"
    return name_part


def format_profit_with_signal(signal_label: str, profit_pln: float, notional_pln: float = 1_000_000) -> str:
    """Format signal with ultra-compact styling.
    
    Apple-quality UX: Clean, scannable, minimal.
    Visual hierarchy:
      - ▲▲▼▼ = Strong signals
      - △▽ = Notable moves
      - ↑↓ = Regular signals
    """
    pct_return = (profit_pln / notional_pln * 100) if notional_pln > 0 else 0.0

    # Compact profit display
    abs_profit = abs(profit_pln)
    if abs_profit >= 1_000_000:
        profit_compact = f"{profit_pln/1_000_000:+.0f}M"
    elif abs_profit >= 1_000:
        profit_compact = f"{profit_pln/1_000:+.0f}k"
    else:
        profit_compact = f"{profit_pln:+.0f}"
    
    if isinstance(signal_label, str):
        label_upper = signal_label.upper()
        
        # EXIT signal: PIT violation triggered (February 2026)
        if label_upper == "EXIT":
            return f"[bold red]EXIT[/bold red]"
        # Strong signals: ▲▲▼▼
        elif label_upper.startswith("STRONG BUY"):
            return f"[bold #00d700]▲▲{pct_return:+.1f}%[/bold #00d700]"
        elif label_upper.startswith("STRONG SELL"):
            return f"[bold indian_red1]▼▼{pct_return:+.1f}%[/bold indian_red1]"
        # Regular BUY/SELL: ↑↓
        elif "SELL" in label_upper:
            return f"[indian_red1]↓{pct_return:+.1f}%[/indian_red1]"
        elif "BUY" in label_upper:
            return f"[#00d700]↑{pct_return:+.1f}%[/#00d700]"
        else:
            # HOLD: Notable moves
            if pct_return > 10.0:
                return f"[bold #00d700]△{pct_return:+.1f}%[/bold #00d700]"
            elif pct_return < -10.0:
                return f"[bold indian_red1]▽{pct_return:+.1f}%[/bold indian_red1]"
            elif pct_return > 3.0:
                return f"[#00d700]△{pct_return:+.1f}%[/#00d700]"
            elif pct_return < -3.0:
                return f"[indian_red1]▽{pct_return:+.1f}%[/indian_red1]"
            else:
                return f"[dim]{pct_return:+.1f}%[/dim]"
    
    return f"{pct_return:+.1f}%"


def extract_symbol_from_title(title: str) -> str:
    """Extract the canonical symbol from a title like 'Company Name (TICKER) — ...'.
    
    Returns empty string if not found.
    Uses the last pair of parentheses to be robust to names containing parentheses.
    """
    try:
        s = str(title)
        # Find last '(' and next ')'
        left_paren = s.rfind('(')
        right_paren = s.find(')', left_paren + 1) if left_paren != -1 else -1
        if left_paren != -1 and right_paren != -1 and right_paren > left_paren + 1:
            token = s[left_paren + 1:right_paren].strip()
            return token.upper()
    except Exception:
        pass
    return ""


def format_risk_temperature(temperature: float) -> str:
    """Format risk temperature with color coding.
    
    Risk Temperature Levels:
        0.0 - 0.5: Normal (green)
        0.5 - 1.0: Elevated (yellow)
        1.0 - 1.5: Stressed (orange)
        1.5 - 2.0: Crisis (red)
    """
    if temperature is None or not isinstance(temperature, (int, float)):
        return "[dim]·[/dim]"
    
    temp = float(temperature)
    
    if temp < 0.3:
        # Very calm - show dot for minimal distraction
        return "[dim]·[/dim]"
    elif temp < 0.5:
        # Normal - subtle green
        return f"[green]{temp:.1f}[/green]"
    elif temp < 1.0:
        # Elevated - yellow warning
        return f"[yellow]⚠{temp:.1f}[/yellow]"
    elif temp < 1.5:
        # Stressed - orange alert
        return f"[orange1]🔥{temp:.1f}[/orange1]"
    else:
        # Crisis - red emergency
        return f"[bold red]🚨{temp:.1f}[/bold red]"


def format_risk_scale_factor(scale: float, temperature: float) -> str:
    """Format risk scale factor showing position reduction.
    
    Shows the effective position multiplier after risk temperature scaling.
    """
    if scale is None or not isinstance(scale, (int, float)):
        return "[dim]1.00[/dim]"
    
    scale = float(scale)
    
    if scale >= 0.95:
        return f"[dim]{scale:.2f}[/dim]"  # Near full - don't distract
    elif scale >= 0.70:
        return f"[yellow]×{scale:.2f}[/yellow]"  # Modest reduction
    elif scale >= 0.40:
        return f"[orange1]×{scale:.2f}[/orange1]"  # Significant reduction
    else:
        return f"[bold red]×{scale:.2f}[/bold red]"  # Severe reduction


# =============================================================================
# BACKWARD COMPATIBILITY: Re-export from canonical temperature modules
# =============================================================================
# Temperature-related rendering is now owned by the temperature modules.
# These re-exports maintain backward compatibility for any code that imports
# from signals_ux.py.
# =============================================================================

# Re-export render_crash_risk_assessment from metals_risk_temperature
try:
    from decision.metals_risk_temperature import render_crash_risk_assessment
except ImportError:
    # Fallback stub if metals module not available
    def render_crash_risk_assessment(*args, **kwargs):
        pass

# Re-export render_risk_temperature_summary from risk_temperature
try:
    from decision.risk_temperature import render_risk_temperature_summary
except ImportError:
    # Fallback stub if risk_temperature module not available
    def render_risk_temperature_summary(*args, **kwargs):
        pass


def render_detailed_signal_table(
    crash_risk_pct: float,
    crash_risk_level: str,
    vol_inversion_count: int = 0,
    inverted_metals: Optional[List[str]] = None,
    momentum_data: Optional[Dict[str, float]] = None,
    console: Console = None,
) -> None:
    """
    Render a comprehensive crash risk assessment panel.
    
    This is a reusable component for displaying crash risk across different views:
    - Main risk temperature summary
    - Tune UX output
    - Signals summary
    
    Design: Apple-quality minimalist display with clear visual hierarchy.
    Premium gauge visualization with actionable guidance.
    
    Args:
        crash_risk_pct: Crash probability (0.0 to 1.0)
        crash_risk_level: Level string ("Low", "Moderate", "Elevated", "High", "Extreme")
        vol_inversion_count: Number of metals with vol term structure inversion
        inverted_metals: List of metal names that show inversion
        momentum_data: Dict of metal_name -> 5d return for momentum display
        console: Rich console instance
    """
    if console is None:
        console = Console()
    
    # Skip if risk is trivial
    if crash_risk_pct <= 0.02:
        return
    
    # Determine styling based on risk level
    CRASH_STYLES = {
        "Extreme": {
            "style": "bold red",
            "label_style": "bold red",
            "border_style": "red",
            "desc": "Imminent correction likely",
            "action": "REDUCE EXPOSURE IMMEDIATELY",
            "action_style": "bold red",
        },
        "High": {
            "style": "bold red",
            "label_style": "bold red",
            "border_style": "red",
            "desc": "Significant downside risk",
            "action": "Consider hedging positions",
            "action_style": "bold yellow",
        },
        "Elevated": {
            "style": "bold yellow",
            "label_style": "bold yellow",
            "border_style": "yellow",
            "desc": "Above-average drawdown risk",
            "action": "Tighten stops, review positions",
            "action_style": "yellow",
        },
        "Moderate": {
            "style": "yellow",
            "label_style": "yellow",
            "border_style": "bright_black",
            "desc": "Mild caution advised",
            "action": "Monitor closely",
            "action_style": "dim",
        },
        "Low": {
            "style": "bright_green",
            "label_style": "bright_green",
            "border_style": "bright_black",
            "desc": "Normal market conditions",
            "action": "Business as usual",
            "action_style": "dim",
        },
    }
    
    style_config = CRASH_STYLES.get(crash_risk_level, CRASH_STYLES["Low"])
    
    # Build the panel content
    lines = []
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # GAUGE VISUALIZATION - Premium semicircular-style risk meter
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Scale labels row
    scale_row = Text()
    scale_row.append("     0%", style="bright_green")
    scale_row.append("        ", style="dim")
    scale_row.append("20%", style="green")
    scale_row.append("        ", style="dim")
    scale_row.append("40%", style="yellow")
    scale_row.append("        ", style="dim")
    scale_row.append("60%", style="red")
    lines.append(scale_row)
    
    # Gauge track with position indicator
    gauge_width = 44
    filled_pos = int(min(1.0, crash_risk_pct / 0.60) * gauge_width)
    
    gauge_row = Text()
    gauge_row.append("     ", style="")  # Left padding
    
    # Build gradient gauge with needle position
    for i in range(gauge_width):
        pct_at_pos = (i / gauge_width) * 0.60
        
        # Determine segment color based on position
        if pct_at_pos < 0.15:
            seg_style = "bright_green"
        elif pct_at_pos < 0.25:
            seg_style = "green"
        elif pct_at_pos < 0.40:
            seg_style = "yellow"
        elif pct_at_pos < 0.50:
            seg_style = "orange1"
        else:
            seg_style = "red"
        
        # Draw filled vs unfilled
        if i < filled_pos:
            gauge_row.append("━", style=f"bold {seg_style}")
        elif i == filled_pos:
            # Needle position - prominent marker
            gauge_row.append("┃", style="bold white")
        else:
            gauge_row.append("─", style="bright_black")
    
    lines.append(gauge_row)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # MAIN RISK READOUT - Large, prominent display
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Percentage and level
    main_row = Text()
    main_row.append("\n     ", style="")
    main_row.append(f"{crash_risk_pct:>5.0%}", style=f"bold {style_config['label_style']}")
    main_row.append("  ", style="")
    main_row.append(f"{crash_risk_level.upper()}", style=style_config['style'])
    lines.append(main_row)
    
    # Description
    desc_row = Text()
    desc_row.append("     ", style="")
    desc_row.append(style_config['desc'], style="dim italic")
    lines.append(desc_row)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # ACTIONABLE GUIDANCE - What to do
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if crash_risk_pct >= 0.20:  # Only show action for elevated+ levels
        action_row = Text()
        action_row.append("\n     ", style="")
        action_row.append("→ ", style="bold white")
        action_row.append(style_config['action'], style=style_config['action_style'])
        lines.append(action_row)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # CONTRIBUTING FACTORS - Vol spike + Momentum
    # ═══════════════════════════════════════════════════════════════════════════════
    
    factors_shown = False
    
    if vol_inversion_count > 0:
        factors_shown = True
        vol_row = Text()
        vol_row.append("\n     ", style="")
        vol_row.append("Vol Spike: ", style="bold yellow")
        if inverted_metals:
            vol_row.append(", ".join(inverted_metals), style="yellow")
        else:
            vol_row.append(f"{vol_inversion_count} metals", style="yellow")
        lines.append(vol_row)
        
        # Compact explanation
        explain_row = Text()
        explain_row.append("       ", style="")
        explain_row.append("Short-term vol exceeding long-term = stress building", style="dim italic")
        lines.append(explain_row)
    
    if momentum_data:
        # Show all metals with any notable move (>= 1%)
        significant_moves = [
            (name, ret) for name, ret in momentum_data.items()
            if abs(ret) >= 0.01  # Show >= 1% moves
        ]
        
        if significant_moves:
            factors_shown = True
            prefix = "\n     " if not vol_inversion_count else "\n     "
            mom_row = Text()
            mom_row.append(prefix, style="")
            mom_row.append("Momentum: ", style="bold cyan")
            
            # Show all metals, sorted by magnitude
            sorted_moves = sorted(significant_moves, key=lambda x: abs(x[1]), reverse=True)
            for idx, (name, ret) in enumerate(sorted_moves):
                if idx > 0:
                    mom_row.append("  ", style="")  # Spacing between items
                arrow = "↑" if ret >= 0 else "↓"
                style = "green" if ret >= 0 else "red"
                mom_row.append(f"{name} {arrow}{ret:+.1%}", style=style)
            
            lines.append(mom_row)
    
    # Combine all lines
    content = Text()
    for i, line in enumerate(lines):
        if i > 0:
            content.append("\n")
        content.append_text(line)
    
    # Create panel with appropriate border style
    panel = Panel(
        content,
        title="[bold]Crash Risk Assessment[/bold]",
        title_align="left",
        border_style=style_config['border_style'],
        box=box.ROUNDED,
        padding=(0, 1),
    )
    
    console.print()
    console.print(panel)
    console.print()
    console.print()


def render_detailed_signal_table(
    asset_symbol: str,
    title: str,
    signals: List,  # List[Signal]
    price_series: pd.Series,
    confidence_level: float,
    used_student_t_mapping: bool,
    show_caption: bool = True
) -> None:
    """Render extraordinary Apple-quality detailed signal analysis table.

    Design Philosophy (Senior Apple UX Professor, 60 years experience):
    ═══════════════════════════════════════════════════════════════════

    1. ALIGNMENT IS SACRED
       - Every column perfectly aligned using Rich Table
       - Monospace precision for numbers
       - Visual rhythm guides the eye

    2. INFORMATION HIERARCHY
       - Primary: Symbol, Price, Signal
       - Secondary: Probability, Return
       - Tertiary: CI, Details

    3. COLOR WITH PURPOSE
       - Green = opportunity (bullish)
       - Red = caution (bearish)
       - Dim = neutral/context
       - White = primary data

    4. WHITESPACE IS DESIGN
       - Breathing room between sections
       - Clean separation with horizontal rules
       - Vertical rhythm through consistent spacing
    
    Args:
        asset_symbol: Trading symbol
        title: Full descriptive title
        signals: List of Signal dataclass instances
        price_series: Historical price series
        confidence_level: Two-sided confidence level (e.g., 0.68)
        used_student_t_mapping: Whether Student-t CDF was used
        show_caption: Whether to show legend
    """
    console = Console(force_terminal=True, width=120)
    last_close = convert_to_float(price_series.iloc[-1])
    last_date = price_series.index[-1].date()
    ci_pct = int(confidence_level * 100)

    # ═══════════════════════════════════════════════════════════════════════════════
    # DETERMINE OVERALL SENTIMENT
    # ═══════════════════════════════════════════════════════════════════════════════

    buy_count = sum(1 for s in signals if "BUY" in (s.label or "").upper())
    sell_count = sum(1 for s in signals if "SELL" in (s.label or "").upper())

    if buy_count >= 5:
        sentiment = ("▲", "bold bright_green", "BULLISH")
    elif sell_count >= 5:
        sentiment = ("▼", "bold indian_red1", "BEARISH")
    elif buy_count > sell_count + 1:
        sentiment = ("△", "bright_green", "LEAN LONG")
    elif sell_count > buy_count + 1:
        sentiment = ("▽", "indian_red1", "LEAN SHORT")
    else:
        sentiment = ("◆", "bright_cyan", "NEUTRAL")

    # ═══════════════════════════════════════════════════════════════════════════════
    # HEADER PANEL - Cinematic Asset Identity
    # ═══════════════════════════════════════════════════════════════════════════════
    
    console.print()

    # Format price intelligently
    if last_close >= 10000:
        price_str = f"{last_close:,.0f}"
    elif last_close >= 100:
        price_str = f"{last_close:,.2f}"
    elif last_close >= 1:
        price_str = f"{last_close:.2f}"
    else:
        price_str = f"{last_close:.4f}"
    
    # Extract company name
    name_part = ""
    if title and " — " in title:
        name_part = title.split(" — ")[0].strip()
    elif title:
        name_part = title.strip()

    # Detect regime from signals
    regime_label = ""
    if signals and hasattr(signals[0], 'regime_label') and signals[0].regime_label:
        regime_label = signals[0].regime_label
    elif signals and hasattr(signals[0], 'regime'):
        regime_map = {0: "LOW_VOL_TREND", 1: "HIGH_VOL_TREND", 2: "LOW_VOL_RANGE",
                      3: "HIGH_VOL_RANGE", 4: "CRISIS_JUMP"}
        regime_label = regime_map.get(signals[0].regime, "")

    regime_colors = {
        "LOW_VOL_TREND": "bright_cyan",
        "HIGH_VOL_TREND": "yellow",
        "LOW_VOL_RANGE": "bright_green",
        "HIGH_VOL_RANGE": "orange1",
        "CRISIS_JUMP": "bold red",
    }
    regime_style = regime_colors.get(regime_label, "dim")

    # Build header content
    header_text = Text()
    header_text.append(f"{sentiment[0]} ", style=sentiment[1])
    header_text.append(asset_symbol, style="bold bright_white")
    if name_part and name_part != asset_symbol:
        header_text.append(f"  {name_part}", style="dim")
    
    # Build subheader with metadata
    sub_text = Text()
    sub_text.append(price_str, style="bold white")
    if regime_label:
        sub_text.append("  │  ", style="dim")
        sub_text.append(regime_label, style=regime_style)
    sub_text.append("  │  ", style="dim")
    sub_text.append(str(last_date), style="dim")
    sub_text.append("  │  ", style="dim")
    cdf_name = "Student-t" if used_student_t_mapping else "Gaussian"
    sub_text.append(cdf_name, style="dim italic")

    # Create elegant header panel
    header_panel = Panel(
        Align.center(Text.assemble(header_text, "\n", sub_text)),
        box=box.HEAVY,
        border_style="bright_blue",
        padding=(0, 2),
    )
    console.print(header_panel)
    console.print()

    # ═══════════════════════════════════════════════════════════════════════════════
    # SIGNAL TABLE - Perfect Alignment with Rich Table
    # ═══════════════════════════════════════════════════════════════════════════════

    table = Table(
        show_header=True,
        header_style="bold white on grey23",
        border_style="bright_blue",
        box=box.SIMPLE_HEAD,
        padding=(0, 1),
        collapse_padding=False,
        show_edge=True,
        row_styles=["", "on grey7"],  # Alternating row colors for readability
    )
    
    # Define columns with precise widths
    table.add_column("Horizon", justify="left", style="bold", width=11, no_wrap=True)
    table.add_column("P(r>0)", justify="right", width=8, no_wrap=True)
    table.add_column("E[return]", justify="right", width=10, no_wrap=True)
    table.add_column(f"CI {ci_pct}%", justify="center", width=20, no_wrap=True)
    table.add_column("Profit", justify="right", width=10, no_wrap=True)
    table.add_column("Signal", justify="center", width=12, no_wrap=True)
    table.add_column("Strength", justify="left", width=12, no_wrap=True)

    for signal in signals:
        notional = 1_000_000
        pct_return = signal.profit_pln / notional * 100
        
        # ─────────────────────────────────────────────────────────────────────────
        # HORIZON - Human readable
        # ─────────────────────────────────────────────────────────────────────────
        horizon_map = {
            1: "1 day", 3: "3 days", 5: "5 days", 7: "1 week",
            14: "2 weeks", 21: "1 month", 42: "6 weeks",
            63: "3 months", 126: "6 months", 252: "12 months"
        }
        horizon_label = horizon_map.get(signal.horizon_days, f"{signal.horizon_days}d")
        
        # ─────────────────────────────────────────────────────────────────────────
        # PROBABILITY - Color coded with precision
        # ─────────────────────────────────────────────────────────────────────────
        p_val = signal.p_up * 100
        if p_val >= 70:
            prob_str = f"[bold bright_green]{p_val:5.1f}%[/]"
        elif p_val >= 58:
            prob_str = f"[bright_green]{p_val:5.1f}%[/]"
        elif p_val <= 30:
            prob_str = f"[bold indian_red1]{p_val:5.1f}%[/]"
        elif p_val <= 42:
            prob_str = f"[indian_red1]{p_val:5.1f}%[/]"
        else:
            prob_str = f"[dim]{p_val:5.1f}%[/]"
        
        # ─────────────────────────────────────────────────────────────────────────
        # EXPECTED RETURN - Percentage format
        # ─────────────────────────────────────────────────────────────────────────
        exp_ret_pct = signal.exp_ret * 100
        if exp_ret_pct >= 5:
            ret_str = f"[bold bright_green]{exp_ret_pct:+7.2f}%[/]"
        elif exp_ret_pct >= 1:
            ret_str = f"[bright_green]{exp_ret_pct:+7.2f}%[/]"
        elif exp_ret_pct <= -5:
            ret_str = f"[bold indian_red1]{exp_ret_pct:+7.2f}%[/]"
        elif exp_ret_pct <= -1:
            ret_str = f"[indian_red1]{exp_ret_pct:+7.2f}%[/]"
        else:
            ret_str = f"[dim]{exp_ret_pct:+7.2f}%[/]"
        
        # ─────────────────────────────────────────────────────────────────────────
        # CONFIDENCE INTERVAL - Clean bracket format
        # ─────────────────────────────────────────────────────────────────────────
        ci_low_pct = signal.ci_low * 100
        ci_high_pct = signal.ci_high * 100
        ci_str = f"[dim][{ci_low_pct:+6.1f}%, {ci_high_pct:+6.1f}%][/]"
        
        # ─────────────────────────────────────────────────────────────────────────
        # PROFIT - Compact with smart units
        # ─────────────────────────────────────────────────────────────────────────
        profit = signal.profit_pln
        if abs(profit) >= 1_000_000:
            profit_str = f"{profit/1_000_000:+.1f}M"
        elif abs(profit) >= 1_000:
            profit_str = f"{profit/1_000:+.0f}k"
        else:
            profit_str = f"{profit:+.0f}"
        
        if pct_return >= 10:
            profit_styled = f"[bold bright_green]{profit_str:>8}[/]"
        elif pct_return >= 2:
            profit_styled = f"[bright_green]{profit_str:>8}[/]"
        elif pct_return <= -10:
            profit_styled = f"[bold indian_red1]{profit_str:>8}[/]"
        elif pct_return <= -2:
            profit_styled = f"[indian_red1]{profit_str:>8}[/]"
        else:
            profit_styled = f"[dim]{profit_str:>8}[/]"
        
        # ─────────────────────────────────────────────────────────────────────────
        # SIGNAL BADGE - Clear visual hierarchy
        # ─────────────────────────────────────────────────────────────────────────
        label_upper = (signal.label or "HOLD").upper()
        if "STRONG" in label_upper and "BUY" in label_upper:
            signal_badge = "[bold bright_green]▲▲ BUY[/]"
        elif "STRONG" in label_upper and "SELL" in label_upper:
            signal_badge = "[bold indian_red1]▼▼ SELL[/]"
        elif "BUY" in label_upper:
            signal_badge = "[bright_green]↑ BUY[/]"
        elif "SELL" in label_upper:
            signal_badge = "[indian_red1]↓ SELL[/indian_red1]"
        else:
            signal_badge = "[dim]— HOLD[/]"

        # ─────────────────────────────────────────────────────────────────────────
        # STRENGTH BAR - True signal strength indicator
        # Based on: distance from neutral (50%) + expected return magnitude
        # This reflects actual model confidence, not just probability
        # ─────────────────────────────────────────────────────────────────────────
        
        # Calculate true signal strength:
        # - Base: distance from 50% (neutral)
        # - Boost: magnitude of expected return
        # - Result: 0-100% strength score
        
        distance_from_neutral = abs(p_val - 50)  # 0-50 scale
        exp_ret_magnitude = abs(exp_ret_pct)  # expected return %
        
        # Combine: probability distance + return magnitude (capped)
        # Weight: 70% probability distance, 30% return magnitude
        strength_score = (distance_from_neutral * 0.7) + min(exp_ret_magnitude * 3, 15) * 0.3
        
        # Convert to 0-10 bar scale
        # strength_score of 0 = 0 bars, 25+ = 10 bars
        bars = min(10, max(0, int(strength_score / 2.5)))
        
        if p_val >= 58:
            # Bullish
            bar_char = "█" * bars + "░" * (10 - bars)
            if p_val >= 70 or (p_val >= 65 and exp_ret_pct >= 3):
                strength_bar = f"[bold bright_green]{bar_char}[/]"
            elif bars >= 3:
                strength_bar = f"[green]{bar_char}[/]"
            else:
                strength_bar = f"[dim green]{bar_char}[/]"
        elif p_val <= 42:
            # Bearish
            bar_char = "█" * bars + "░" * (10 - bars)
            if p_val <= 30 or (p_val <= 35 and exp_ret_pct <= -3):
                strength_bar = f"[bold indian_red1]{bar_char}[/]"
            elif bars >= 3:
                strength_bar = f"[red]{bar_char}[/]"
            else:
                strength_bar = f"[dim red]{bar_char}[/]"
        else:
            # Neutral - show minimal strength
            strength_bar = f"[dim]{'─' * 10}[/]"

        # Add the row
        table.add_row(
            horizon_label,
            prob_str,
            ret_str,
            ci_str,
            profit_styled,
            signal_badge,
            strength_bar,
        )

    console.print(table)
    console.print()

    # ═══════════════════════════════════════════════════════════════════════════════
    # LEGEND FOOTER - Contextual help
    # ═══════════════════════════════════════════════════════════════════════════════

    if show_caption:
        legend_table = Table.grid(padding=(0, 3))
        legend_table.add_column(justify="left")
        legend_table.add_column(justify="left")
        legend_table.add_column(justify="left")

        legend_table.add_row(
            Text.assemble(("P(r>0)", "bold dim"), (" prob of positive return", "dim")),
            Text.assemble(("E[return]", "bold dim"), (" expected return", "dim")),
            Text.assemble(("Profit", "bold dim"), (" on 1M PLN notional", "dim")),
        )
        console.print(Align.center(legend_table))
        console.print()

        threshold_table = Table.grid(padding=(0, 4))
        threshold_table.add_column(justify="center")
        threshold_table.add_column(justify="center")
        threshold_table.add_column(justify="center")

        threshold_table.add_row(
            Text.assemble(("↑ BUY", "bright_green"), ("  P ≥ 58%", "dim")),
            Text.assemble(("— HOLD", "dim"), ("  42% < P < 58%", "dim")),
            Text.assemble(("↓ SELL", "indian_red1"), ("  P ≤ 42%", "dim")),
        )
        console.print(Align.center(threshold_table))

    console.print()


def extract_market_context(features: Dict[str, pd.Series]) -> tuple[str, str, str]:
    """Extract simple market context descriptions from computed features.
    
    Args:
        features: Dictionary of computed technical features
        
    Returns:
        Tuple of (trend_description, momentum_description, volatility_description)
    """
    trend = convert_to_float(features.get("slow_trend", float("nan")))
    moms = [
        convert_to_float(features.get("mom_21", float("nan"))),
        convert_to_float(features.get("mom_42", float("nan"))),
        convert_to_float(features.get("mom_63", float("nan"))),
        convert_to_float(features.get("mom_126", float("nan"))),
    ]
    vol_reg = convert_to_float(features.get("vol_regime", float("nan")))

    # Trend assessment
    if np.isfinite(trend) and trend > 0.5:
        trend_desc = "Uptrend"
    elif np.isfinite(trend) and trend < -0.5:
        trend_desc = "Downtrend"
    else:
        trend_desc = "Sideways"

    # Momentum agreement
    valid_moms = [m for m in moms if np.isfinite(m)]
    if valid_moms:
        positive_count = sum(1 for m in valid_moms if m > 0)
        if positive_count >= 3:
            momentum_desc = "Momentum mostly positive"
        elif positive_count <= 1:
            momentum_desc = "Momentum mostly negative"
        else:
            momentum_desc = "Momentum mixed"
    else:
        momentum_desc = "Momentum unclear"

    # Volatility regime
    if np.isfinite(vol_reg) and vol_reg > 1.5:
        volatility_desc = "High volatility (signals weaker)"
    elif np.isfinite(vol_reg) and vol_reg < 0.8:
        volatility_desc = "Calm volatility (signals stronger)"
    else:
        volatility_desc = "Normal volatility"

    return trend_desc, momentum_desc, volatility_desc


def render_augmentation_layers_summary(
    signal,  # Signal dataclass
    console: Console = None,
) -> None:
    """Render compact augmentation layer summary for a signal.
    
    Shows which augmentation layers (Hansen-λ, EVT/GPD, Contaminated-t) are active
    for the current signal computation.
    
    Args:
        signal: Signal dataclass with augmentation layer fields
        console: Rich Console for output
    """
    if console is None:
        console = Console(force_terminal=True, width=120)
    
    # Check if any augmentation layer is active
    hansen_active = getattr(signal, 'hansen_enabled', False)
    evt_active = getattr(signal, 'evt_enabled', False)
    cst_active = getattr(signal, 'cst_enabled', False)
    
    if not (hansen_active or evt_active or cst_active):
        return  # No augmentation layers to display
    
    from rich.text import Text
    
    aug_text = Text()
    aug_text.append("  ", style="dim")
    aug_text.append("Augmentation", style="dim")
    aug_text.append("  ", style="dim")
    
    layers = []
    
    # Hansen Skew-t
    if hansen_active:
        hansen_lambda = getattr(signal, 'hansen_lambda', None)
        hansen_skew = getattr(signal, 'hansen_skew_direction', 'symmetric')
        if hansen_lambda is not None:
            skew_symbol = "←" if hansen_skew == "left" else ("→" if hansen_skew == "right" else "○")
            layers.append(f"[cyan]Hλ={hansen_lambda:+.2f}{skew_symbol}[/cyan]")
        else:
            layers.append("[cyan]Hλ[/cyan]")
    
    # EVT/GPD
    if evt_active:
        evt_xi = getattr(signal, 'evt_xi', None)
        if evt_xi is not None:
            if evt_xi > 0.2:
                tail_label = "heavy"
                tail_style = "indian_red1"
            elif evt_xi > 0.05:
                tail_label = "moderate"
                tail_style = "yellow"
            else:
                tail_label = "light"
                tail_style = "bright_green"
            layers.append(f"[{tail_style}]EVT(ξ={evt_xi:.2f},{tail_label})[/{tail_style}]")
        else:
            layers.append("[indian_red1]EVT[/indian_red1]")
    
    # Contaminated Student-t
    if cst_active:
        cst_epsilon = getattr(signal, 'cst_epsilon', None)
        cst_nu_normal = getattr(signal, 'cst_nu_normal', None)
        cst_nu_crisis = getattr(signal, 'cst_nu_crisis', None)
        if cst_epsilon is not None and cst_nu_crisis is not None:
            layers.append(f"[yellow]CST(ε={cst_epsilon:.0%},ν={cst_nu_crisis:.0f})[/yellow]")
        else:
            layers.append("[yellow]CST[/yellow]")
    
    if layers:
        # Print with markup enabled - use console.print directly with markup=True
        aug_line = "  [dim]Augmentation[/dim]  " + "  ".join(layers)
        console.print(aug_line, markup=True)


def generate_signal_explanation(
    edge_score: float,
    probability_up: float,
    trend_description: str,
    momentum_description: str,
    volatility_description: str
) -> str:
    """Generate plain English explanation for a signal.
    
    Args:
        edge_score: Risk-adjusted edge (z-score)
        probability_up: Probability of positive return
        trend_description: Market trend description
        momentum_description: Momentum description
        volatility_description: Volatility regime description
        
    Returns:
        Plain English explanation combining all factors
    """
    # Strength descriptor
    strength = "slight"
    if abs(edge_score) >= 1.5:
        strength = "strong"
    elif abs(edge_score) >= 0.7:
        strength = "moderate"
    
    direction = "rise" if edge_score >= 0 else "fall"
    
    return (
        f"{trend_description}. {momentum_description}. {volatility_description}. "
        f"About {probability_up*100:.0f}% chance of a {direction}. Confidence {strength}."
    )


def render_simplified_signal_table(
    asset_symbol: str,
    title: str,
    signals: List,  # List[Signal]
    price_series: pd.Series,
    features: Dict[str, pd.Series]
) -> List[str]:
    """Render simplified signal table with plain English explanations.
    
    Args:
        asset_symbol: Trading symbol
        title: Full descriptive title
        signals: List of Signal dataclass instances
        price_series: Historical price series
        features: Dictionary of computed technical features
        
    Returns:
        List of explanation strings for each signal
    """
    console = Console()
    last_close = convert_to_float(price_series.iloc[-1])
    trend_desc, momentum_desc, volatility_desc = extract_market_context(features)

    table = Table(
        title=f"[bold]{asset_symbol}[/bold] — {title}\n[dim]Last: {last_close:,.4f} on {price_series.index[-1].date()}[/dim]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        row_styles=["", "dim"],
    )
    table.add_column("Timeframe", justify="left", width=12)
    table.add_column("Chance ↑", justify="right", width=10)
    table.add_column("Signal", justify="center", width=15)
    table.add_column("Summary", justify="left")
    table.caption = "[dim]BUY = expect price to rise · SELL = expect price to fall · HOLD = uncertain[/dim]"

    explanations: List[str] = []
    for signal in signals:
        explanation = generate_signal_explanation(
            signal.score, signal.p_up, trend_desc, momentum_desc, volatility_desc
        )
        explanations.append(explanation)
        
        # Format probability with color (pleasant, non-neon colors)
        p_val = signal.p_up * 100
        if p_val >= 60:
            p_cell = f"[#00d700]{p_val:.0f}%[/#00d700]"
        elif p_val <= 40:
            p_cell = f"[indian_red1]{p_val:.0f}%[/indian_red1]"
        else:
            p_cell = f"[#9e9e9e]{p_val:.0f}%[/#9e9e9e]"
        
        # Format signal label with styling (pleasant, non-neon colors)
        label_upper = signal.label.upper() if isinstance(signal.label, str) else ""
        if "STRONG" in label_upper and "BUY" in label_upper:
            label_cell = f"[bold #00d700]⬆ BUY[/bold #00d700]"
        elif "STRONG" in label_upper and "SELL" in label_upper:
            label_cell = f"[bold indian_red1]⬇ SELL[/bold indian_red1]"
        elif "BUY" in label_upper:
            label_cell = f"[#00d700]BUY[/#00d700]"
        elif "SELL" in label_upper:
            label_cell = f"[indian_red1]SELL[/indian_red1]"
        else:
            label_cell = f"[#9e9e9e]HOLD[/#9e9e9e]"
        
        table.add_row(
            format_horizon_label(signal.horizon_days),
            p_cell,
            label_cell,
            f"[dim]{explanation}[/dim]",
        )

    console.print(table)
    console.print()
    return explanations


def render_multi_asset_summary_table(summary_rows: List[Dict], horizons: List[int], title_override: str = None, asset_col_width: int = None, console: Console = None) -> None:
    """Render world-class compact signal heatmap table.
    
    Split into two tables:
    1. Active signals (non-EXIT) - sorted alphabetically by Asset
    2. EXIT signals (belief withdrawn) - sorted alphabetically by Asset
    """
    if not summary_rows:
        return

    # Separate rows into EXIT and non-EXIT categories
    exit_rows = []
    active_rows = []
    
    for row in summary_rows:
        horizon_signals = row.get("horizon_signals", {})
        # Check if ANY horizon has EXIT signal
        has_exit = False
        for horizon in horizons:
            signal_data = horizon_signals.get(horizon) or horizon_signals.get(str(horizon)) or {}
            label = signal_data.get("label", "HOLD")
            if str(label).upper() == "EXIT":
                has_exit = True
                break
        
        if has_exit:
            exit_rows.append(row)
        else:
            active_rows.append(row)
    
    # Sort both lists alphabetically by asset label
    def asset_sort_key(row: Dict) -> str:
        label = row.get("asset_label", "")
        # Extract just the company name for sorting (before the ticker)
        if isinstance(label, str):
            # Remove Rich markup tags
            import re
            plain = re.sub(r"\[/?[^\]]+\]", "", label)
            return plain.lower()
        return ""
    
    active_rows = sorted(active_rows, key=asset_sort_key)
    exit_rows = sorted(exit_rows, key=asset_sort_key)

    # Compact asset column width
    import re
    def _plain_len(text: str) -> int:
        if not isinstance(text, str):
            return 0
        return len(re.sub(r"\[/?[^\]]+\]", "", text))
    if asset_col_width is None:
        longest_asset = max((_plain_len(r.get("asset_label", "")) for r in summary_rows), default=0)
        asset_col_width = max(40, min(52, longest_asset + 4))

    if console is None:
        console = Console(force_terminal=True, width=200)
    
    # Helper function to render a single table
    def _render_table(rows: List[Dict], table_title: str, title_style: str, border_style: str) -> None:
        if not rows:
            return
            
        table = Table(
            title=table_title,
            title_style=title_style,
            show_header=True,
            header_style="bold white",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(0, 1),
            collapse_padding=False,
            row_styles=["", "on grey7"],
        )
        # Asset column
        table.add_column("Asset", justify="left", style="white", width=asset_col_width, no_wrap=True, overflow="ellipsis")
        # Crash Risk column (0-100 momentum-based multi-factor)
        table.add_column("Crash", justify="right", width=5, style="red")
        # Exhaustion columns
        table.add_column("↑", justify="right", width=3, style="indian_red1")
        table.add_column("↓", justify="right", width=3, style="bright_green")
        
        # Horizon columns
        horizon_labels = {1: "1d", 3: "3d", 7: "1w", 21: "1m", 63: "3m", 126: "6m", 252: "12m"}
        for horizon in horizons:
            label = horizon_labels.get(horizon, f"{horizon}d")
            table.add_column(label, justify="center", width=11, no_wrap=True)

        for row in rows:
            asset_label = row.get("asset_label", "Unknown")
            horizon_signals = row.get("horizon_signals", {})
            
            # Get crash risk score (0-100) and format display
            crash_risk_score = row.get("crash_risk_score", 0)
            if crash_risk_score is None:
                crash_risk_score = 0
            crash_risk_score = int(crash_risk_score)
            
            # Format crash risk display with intuitive color coding
            # Low risk = calm colors, High risk = alarming colors
            if crash_risk_score < 15:
                # Very low risk - dim/invisible
                crash_risk_display = f"[dim]·[/dim]"
            elif crash_risk_score < 30:
                # Low risk - calm green
                crash_risk_display = f"[bright_green]{crash_risk_score}[/bright_green]"
            elif crash_risk_score < 50:
                # Moderate risk - yellow warning
                crash_risk_display = f"[yellow]{crash_risk_score}[/yellow]"
            elif crash_risk_score < 70:
                # Elevated risk - orange alert
                crash_risk_display = f"[orange1]{crash_risk_score}[/orange1]"
            elif crash_risk_score < 85:
                # High risk - red
                crash_risk_display = f"[red]{crash_risk_score}[/red]"
            else:
                # Extreme risk - bold red with emphasis
                crash_risk_display = f"[bold red]{crash_risk_score}[/bold red]"
            
            # Compute max UE↑ and UE↓
            max_ue_up = 0.0
            max_ue_down = 0.0
            for horizon in horizons:
                signal_data = horizon_signals.get(horizon) or horizon_signals.get(str(horizon)) or {}
                ue_up = signal_data.get("ue_up", 0.0) or 0.0
                ue_down = signal_data.get("ue_down", 0.0) or 0.0
                if ue_up > max_ue_up:
                    max_ue_up = ue_up
                if ue_down > max_ue_down:
                    max_ue_down = ue_down
            
            # Format UE↑
            ue_up_pct = int(max_ue_up * 100)
            if ue_up_pct < 5:
                ue_up_display = "[dim]·[/dim]"
            elif max_ue_up >= 0.6:
                ue_up_display = f"[indian_red1]{ue_up_pct}[/indian_red1]"
            elif max_ue_up >= 0.3:
                ue_up_display = f"[yellow]{ue_up_pct}[/yellow]"
            else:
                ue_up_display = f"[dim]{ue_up_pct}[/dim]"

            # Format UE↓
            ue_down_pct = int(max_ue_down * 100)
            if ue_down_pct < 5:
                ue_down_display = "[dim]·[/dim]"
            elif max_ue_down >= 0.6:
                ue_down_display = f"[#00d700]{ue_down_pct}[/#00d700]"
            elif max_ue_down >= 0.3:
                ue_down_display = f"[cyan]{ue_down_pct}[/cyan]"
            else:
                ue_down_display = f"[dim]{ue_down_pct}[/dim]"
            
            cells = []
            for horizon in horizons:
                signal_data = horizon_signals.get(horizon) or horizon_signals.get(str(horizon)) or {}
                label = signal_data.get("label", "HOLD")
                profit_pln = signal_data.get("profit_pln", 0.0)
                cells.append(format_profit_with_signal(label, profit_pln))
            
            table.add_row(asset_label, crash_risk_display, ue_up_display, ue_down_display, *cells)

        console.print(table)
        console.print()
    
    # Render active signals table first (if any)
    if active_rows:
        active_title = title_override if title_override else f"Active Signals ({len(active_rows)} assets)"
        _render_table(active_rows, active_title, "bold bright_white", "dim")
    
    # Render EXIT signals table (if any)
    if exit_rows:
        exit_title = f"EXIT — Belief Withdrawn ({len(exit_rows)} assets)"
        _render_table(exit_rows, exit_title, "bold red", "red")


def render_sector_summary_tables(summary_rows: List[Dict], horizons: List[int]) -> None:
    """Render extraordinary Apple-quality sector-grouped signal tables."""
    if not summary_rows:
        return

    buckets: Dict[str, List[Dict]] = {}
    for row in summary_rows:
        sector = row.get("sector", "") or ""
        sector = sector.strip() if sector else "Other"
        buckets.setdefault(sector, []).append(row)

    import re
    from datetime import datetime
    from rich.rule import Rule
    from rich.align import Align
    
    def _plain_len(text: str) -> int:
        if not isinstance(text, str):
            return 0
        return len(re.sub(r"\[/?[^\]]+\]", "", text))
    longest_asset = max((_plain_len(r.get("asset_label", "")) for r in summary_rows), default=0)
    asset_col_width = max(38, min(50, longest_asset + 4))

    console = Console(force_terminal=True, width=180)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # CINEMATIC HEADER
    # ═══════════════════════════════════════════════════════════════════════════════
    console.print()
    console.print()
    
    title_content = Text()
    title_content.append("\n", style="")
    title_content.append("S I G N A L S", style="bold bright_white")
    title_content.append("\n", style="")
    title_content.append("Bayesian Forecast Dashboard", style="dim")
    title_content.append("\n", style="")
    
    title_panel = Panel(
        Align.center(title_content),
        box=box.DOUBLE,
        border_style="bright_cyan",
        padding=(0, 6),
    )
    console.print(Align.center(title_panel, width=50))
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # LEGEND
    # ═══════════════════════════════════════════════════════════════════════════════
    console.print(Rule(style="dim"))
    console.print()
    
    legend_grid = Table.grid(padding=(0, 2))
    legend_grid.add_column(justify="center")
    legend_grid.add_column(justify="center")
    legend_grid.add_column(justify="center")
    legend_grid.add_column(justify="center")
    legend_grid.add_column(justify="center")
    
    legend_grid.add_row(
        Text.assemble(("▲▲", "bold bright_green"), (" Strong Buy", "dim")),
        Text.assemble(("▼▼", "bold indian_red1"), (" Strong Sell", "dim")),
        Text.assemble(("△▽", "bold white"), (" Notable", "dim")),
        Text.assemble(("↑↓", "bold white"), (" Signal", "dim")),
        Text.assemble(("1M PLN", "bold cyan"), (" notional", "dim")),
    )
    console.print(Align.center(legend_grid))
    console.print()
    
    exhaust_legend = Table.grid(padding=(0, 4))
    exhaust_legend.add_column(justify="center")
    exhaust_legend.add_column(justify="center")
    exhaust_legend.add_column(justify="center")
    
    exhaust_legend.add_row(
        Text.assemble(("CR", "bold yellow"), (" Crash Risk (0-100)", "dim")),
        Text.assemble(("↑%", "bold indian_red1"), (" Overbought (above EMA)", "dim")),
        Text.assemble(("↓%", "bold bright_green"), (" Oversold (below EMA)", "dim")),
    )
    console.print(Align.center(exhaust_legend))
    console.print()
    console.print(Rule(style="dim"))

    # Sort sectors alphabetically but put "Other" at the end
    def sector_sort_key(item):
        sector_name = item[0]
        if sector_name in ("Other", "Unspecified"):
            return ("~", sector_name)
        return ("", sector_name)

    sector_num = 0
    for sector, rows in sorted(buckets.items(), key=sector_sort_key):
        sector_num += 1
        
        # Sector header - format:  14     Defense & Aerospace   (108 assets)
        console.print()
        sector_header = Text()
        sector_header.append(f"  {sector_num:2d}  ", style="bold bright_white on bright_blue")
        sector_header.append("   ", style="")
        sector_header.append(sector, style="bold white")
        sector_header.append("   ", style="")
        sector_header.append(f"({len(rows)} assets)", style="dim")
        console.print(sector_header)
        console.print()
        
        render_multi_asset_summary_table(rows, horizons, title_override=None, asset_col_width=asset_col_width, console=console)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # COMPLETION FOOTER - Centered panel matching header style
    # ═══════════════════════════════════════════════════════════════════════════════
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    
    total_signals = sum(
        1 for row in summary_rows
        for sig in row.get("horizon_signals", {}).values()
        if sig.get("label", "HOLD") not in ("HOLD", "")
    )
    
    # Centered panel footer
    content = Text(justify="center")
    content.append("\n", style="")
    content.append("✓ ", style="bold bright_green")
    content.append("Complete", style="bold white")
    content.append("\n\n", style="")
    content.append(f"{len(summary_rows):,}", style="bold bright_cyan")
    content.append(" assets", style="dim")
    content.append("  ·  ", style="dim")
    content.append(f"{sector_num}", style="bold white")
    content.append(" sectors", style="dim")
    content.append("  ·  ", style="dim")
    content.append(f"{total_signals:,}", style="bold bright_green")
    content.append(" signals", style="dim")
    content.append("\n", style="")
    
    completion_panel = Panel(
        Align.center(content),
        box=box.ROUNDED,
        border_style="bright_green",
        padding=(0, 4),
        width=54,
    )
    console.print(Align.center(completion_panel))
    console.print()


def render_strong_signals_summary(summary_rows: List[Dict], horizons: List[int] = None) -> None:
    """Render high-conviction signal summary tables for short-term trading.
    
    Shows two tables:
    1. STRONG BUY signals (P >= 62%) for next 7 days
    2. STRONG SELL signals (P <= 38%) for next 7 days
    
    Design Philosophy:
    - Focus on actionable short-term signals
    - High conviction only (clear edge)
    - Sorted by signal strength
    
    Args:
        summary_rows: List of asset summary dictionaries from signal generation
        horizons: Optional list of horizons to include (default: [1, 3, 7])
    """
    if not summary_rows:
        return
    
    if horizons is None:
        horizons = [1, 3, 7]  # Focus on short-term: 1d, 3d, 1w
    
    console = Console(force_terminal=True, width=180)
    
    # Thresholds for "high conviction"
    BUY_THRESHOLD = 0.62   # P(r>0) >= 62% for strong buy
    SELL_THRESHOLD = 0.38  # P(r>0) <= 38% for strong sell
    
    # Collect strong signals
    strong_buys = []
    strong_sells = []
    
    for row in summary_rows:
        asset_label = row.get("asset_label", "Unknown")
        horizon_signals = row.get("horizon_signals", {})
        sector = row.get("sector", "Other")
        
        for horizon in horizons:
            signal_data = horizon_signals.get(horizon) or horizon_signals.get(str(horizon)) or {}
            p_up = signal_data.get("p_up", 0.5)
            exp_ret = signal_data.get("exp_ret", 0.0)
            profit_pln = signal_data.get("profit_pln", 0.0)
            label = signal_data.get("label", "HOLD")
            
            # Skip EXIT signals - these indicate untrusted beliefs
            # EXIT signals should not appear in trading recommendations
            if label.upper() == "EXIT":
                continue
            
            # Calculate strength score for sorting
            distance_from_neutral = abs(p_up - 0.5)
            strength = distance_from_neutral + abs(exp_ret) * 0.5
            
            if p_up >= BUY_THRESHOLD:
                strong_buys.append({
                    "asset": asset_label,
                    "sector": sector,
                    "horizon": horizon,
                    "p_up": p_up,
                    "exp_ret": exp_ret,
                    "profit_pln": profit_pln,
                    "strength": strength,
                })
            elif p_up <= SELL_THRESHOLD:
                strong_sells.append({
                    "asset": asset_label,
                    "sector": sector,
                    "horizon": horizon,
                    "p_up": p_up,
                    "exp_ret": exp_ret,
                    "profit_pln": profit_pln,
                    "strength": strength,
                })
    
    # Sort by strength descending
    strong_buys.sort(key=lambda x: x["strength"], reverse=True)
    strong_sells.sort(key=lambda x: x["strength"], reverse=True)
    
    # Limit to top 20 each
    strong_buys = strong_buys[:20]
    strong_sells = strong_sells[:20]
    
    # Only render if we have signals
    if not strong_buys and not strong_sells:
        return
    
    console.print()
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # HEADER - Centered panel
    # ═══════════════════════════════════════════════════════════════════════════════
    header_content = Text(justify="center")
    header_content.append("\n", style="")
    header_content.append("HIGH CONVICTION SIGNALS", style="bold bright_white")
    header_content.append("\n", style="")
    header_content.append("Short-term opportunities (next 7 trading days)", style="dim")
    header_content.append("\n", style="")
    
    header_panel = Panel(
        Align.center(header_content),
        box=box.HEAVY,
        border_style="bright_yellow",
        padding=(0, 2),
        width=54,
    )
    console.print(Align.center(header_panel))
    console.print()
    
    horizon_labels = {1: "1 day", 3: "3 days", 7: "1 week"}
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # STRONG BUY TABLE
    # ═══════════════════════════════════════════════════════════════════════════════
    if strong_buys:
        console.print()
        buy_header = Text()
        buy_header.append("  ▲▲ ", style="bold bright_green")
        buy_header.append("STRONG BUY SIGNALS", style="bold bright_green")
        buy_header.append(f"  ({len(strong_buys)} opportunities)", style="dim")
        console.print(buy_header)
        console.print()
        
        buy_table = Table(
            show_header=True,
            header_style="bold white on green",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 1),
            row_styles=["", "on grey7"],
        )
        
        buy_table.add_column("Asset", justify="left", width=50, no_wrap=True)
        buy_table.add_column("Sector", justify="left", width=30, no_wrap=True, style="dim")
        buy_table.add_column("Horizon", justify="center", width=10)
        buy_table.add_column("P(r>0)", justify="right", width=8)
        buy_table.add_column("E[return]", justify="right", width=10)
        buy_table.add_column("Profit", justify="right", width=10)
        buy_table.add_column("Strength", justify="left", width=12)
        
        for sig in strong_buys:
            horizon_label = horizon_labels.get(sig["horizon"], f"{sig['horizon']}d")
            p_pct = sig["p_up"] * 100
            exp_ret_pct = sig["exp_ret"] * 100
            
            # Profit formatting
            profit = sig["profit_pln"]
            if abs(profit) >= 1_000_000:
                profit_str = f"{profit/1_000_000:+.1f}M"
            elif abs(profit) >= 1_000:
                profit_str = f"{profit/1_000:+.0f}k"
            else:
                profit_str = f"{profit:+.0f}"
            
            # Strength bar
            bars = min(10, max(1, int(sig["strength"] * 40)))
            bar_str = "█" * bars + "░" * (10 - bars)
            
            buy_table.add_row(
                sig["asset"],
                sig["sector"][:28] if sig["sector"] else "",
                horizon_label,
                f"[bold bright_green]{p_pct:.1f}%[/]",
                f"[bright_green]{exp_ret_pct:+.2f}%[/]",
                f"[bright_green]{profit_str}[/]",
                f"[green]{bar_str}[/]",
            )
        
        console.print(buy_table)
        console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # STRONG SELL TABLE
    # ═══════════════════════════════════════════════════════════════════════════════
    if strong_sells:
        console.print()
        sell_header = Text()
        sell_header.append("  ▼▼ ", style="bold indian_red1")
        sell_header.append("STRONG SELL SIGNALS", style="bold indian_red1")
        sell_header.append(f"  ({len(strong_sells)} warnings)", style="dim")
        console.print(sell_header)
        console.print()
        
        sell_table = Table(
            show_header=True,
            header_style="bold white on red",
            border_style="red",
            box=box.ROUNDED,
            padding=(0, 1),
            row_styles=["", "on grey7"],
        )
        
        sell_table.add_column("Asset", justify="left", width=50, no_wrap=True)
        sell_table.add_column("Sector", justify="left", width=30, no_wrap=True, style="dim")
        sell_table.add_column("Horizon", justify="center", width=10)
        sell_table.add_column("P(r>0)", justify="right", width=8)
        sell_table.add_column("E[return]", justify="right", width=10)
        sell_table.add_column("Profit", justify="right", width=10)
        sell_table.add_column("Strength", justify="left", width=12)
        
        for sig in strong_sells:
            horizon_label = horizon_labels.get(sig["horizon"], f"{sig['horizon']}d")
            p_pct = sig["p_up"] * 100
            exp_ret_pct = sig["exp_ret"] * 100
            
            # Profit formatting
            profit = sig["profit_pln"]
            if abs(profit) >= 1_000_000:
                profit_str = f"{profit/1_000_000:+.1f}M"
            elif abs(profit) >= 1_000:
                profit_str = f"{profit/1_000:+.0f}k"
            else:
                profit_str = f"{profit:+.0f}"
            
            # Strength bar
            bars = min(10, max(1, int(sig["strength"] * 40)))
            bar_str = "█" * bars + "░" * (10 - bars)
            
            sell_table.add_row(
                sig["asset"],
                sig["sector"][:28] if sig["sector"] else "",
                horizon_label,
                f"[bold indian_red1]{p_pct:.1f}%[/]",
                f"[indian_red1]{exp_ret_pct:+.2f}%[/]",
                f"[indian_red1]{profit_str}[/]",
                f"[red]{bar_str}[/]",
            )
        
        sell_table.add_row()
        console.print(sell_table)
        console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════════════════════════════════
    console.print()
    footer = Text(justify="center")
    footer.append("Thresholds: ", style="dim")
    footer.append("BUY ", style="bright_green")
    footer.append(f"P ≥ {BUY_THRESHOLD*100:.0f}%", style="dim")
    footer.append("  ·  ", style="dim")
    footer.append("SELL ", style="indian_red1")
    footer.append(f"P ≤ {SELL_THRESHOLD*100:.0f}%", style="dim")
    footer.append("  ·  ", style="dim")
    footer.append("Sorted by signal strength", style="dim italic")
    console.print(Align.center(footer))
    console.print()


def render_portfolio_allocation_table(
    portfolio_result: Dict,
    horizon_days: int,
    notional_pln: float = 1_000_000.0
) -> None:
    """
    Render portfolio allocation table showing optimal weights.
    
    Displays:
    - Asset allocations (EU-based weights)
    - Expected returns per asset
    - Portfolio-level metrics (leverage, diversification, Sharpe)
    - Correlation matrix
    - Comparison with equal-weight allocation
    
    Args:
        portfolio_result: Dictionary from build_multi_asset_portfolio()
        horizon_days: Forecast horizon in trading days
        notional_pln: Notional amount in PLN for position sizing
    """
    console = Console()
    
    # Extract data
    asset_names = portfolio_result["asset_names"]
    weights_kelly = portfolio_result["weights_clamped"]
    expected_returns = portfolio_result["expected_returns"]
    correlation_matrix = portfolio_result["correlation_matrix"]
    portfolio_stats = portfolio_result["portfolio_stats"]
    leverage = portfolio_result["leverage"]
    div_ratio = portfolio_result["diversification_ratio"]
    
    # Build allocation table
    alloc_table = Table(
        title=f"📊 Portfolio Allocation ({format_horizon_label(horizon_days)} horizon)",
        show_header=True,
        header_style="bold cyan"
    )
    alloc_table.add_column("Asset", justify="left", style="bold")
    alloc_table.add_column("Weight (EU)", justify="right")
    alloc_table.add_column("Position (PLN)", justify="right")
    alloc_table.add_column(f"E[Return] ({horizon_days}d)", justify="right")
    alloc_table.add_column("Equal-Weight", justify="right", style="dim")
    
    # Equal weight for comparison
    n_assets = len(asset_names)
    equal_weight = 1.0 / n_assets if n_assets > 0 else 0.0
    
    for i, name in enumerate(asset_names):
        weight = weights_kelly[i]
        position_pln = weight * notional_pln
        exp_ret = expected_returns[i]
        
        # Color code weights (pleasant, non-neon colors)
        if weight > 0.15:
            weight_str = f"[bold #00d700]{weight:+7.2%}[/bold #00d700]"
        elif weight > 0.05:
            weight_str = f"[#00d700]{weight:+7.2%}[/#00d700]"
        elif weight < -0.05:
            weight_str = f"[indian_red1]{weight:+7.2%}[/indian_red1]"
        else:
            weight_str = f"[#9e9e9e]{weight:+7.2%}[/#9e9e9e]"
        
        # Position size formatting
        if abs(position_pln) >= 1_000_000:
            pos_str = f"{position_pln:+,.0f}"
        else:
            pos_str = f"{position_pln:+,.0f}"
        
        # Expected return formatting (pleasant, non-neon colors)
        ret_str = f"{exp_ret:+.4f}"
        if exp_ret > 0.01:
            ret_str = f"[#00d700]{ret_str}[/#00d700]"
        elif exp_ret < -0.01:
            ret_str = f"[indian_red1]{ret_str}[/indian_red1]"
        else:
            ret_str = f"[#9e9e9e]{ret_str}[/#9e9e9e]"
        
        alloc_table.add_row(
            name,
            weight_str,
            pos_str,
            ret_str,
            f"{equal_weight:.2%}"
        )
    
    console.print(alloc_table)
    
    # Portfolio metrics table
    metrics_table = Table(
        title="📈 Portfolio Metrics",
        show_header=True,
        header_style="bold magenta"
    )
    metrics_table.add_column("Metric", justify="left", style="cyan")
    metrics_table.add_column("Value", justify="right")
    metrics_table.add_column("Interpretation", justify="left", style="dim")
    
    # Leverage
    lev_color = "yellow" if leverage > 0.8 else "green"
    lev_interp = "High" if leverage > 0.8 else ("Moderate" if leverage > 0.5 else "Conservative")
    metrics_table.add_row(
        "Leverage (Σ|w_i|)",
        f"[{lev_color}]{leverage:.3f}[/{lev_color}]",
        lev_interp
    )
    
    # Diversification ratio
    div_color = "green" if div_ratio < 0.8 else ("yellow" if div_ratio < 0.95 else "red")
    div_interp = "Well diversified" if div_ratio < 0.8 else ("Moderate diversification" if div_ratio < 0.95 else "Low diversification")
    metrics_table.add_row(
        "Diversification Ratio",
        f"[{div_color}]{div_ratio:.3f}[/{div_color}]",
        div_interp + " (lower = better)"
    )
    
    # Portfolio expected return
    port_ret = portfolio_stats["expected_return"]
    ret_color = "green" if port_ret > 0 else "red"
    metrics_table.add_row(
        f"Expected Return ({horizon_days}d)",
        f"[{ret_color}]{port_ret:+.6f}[/{ret_color}]",
        f"≈ {port_ret * 100:.3f}% over horizon"
    )
    
    # Portfolio volatility
    port_vol = portfolio_stats["volatility"]
    metrics_table.add_row(
        f"Portfolio Volatility ({horizon_days}d)",
        f"{port_vol:.6f}",
        f"≈ {port_vol * 100:.3f}% std dev"
    )
    
    # Sharpe ratio
    sharpe = portfolio_stats["sharpe_ratio"]
    sharpe_color = "green" if sharpe > 1.0 else ("yellow" if sharpe > 0.5 else "red")
    sharpe_interp = "Excellent" if sharpe > 1.5 else ("Good" if sharpe > 1.0 else ("Fair" if sharpe > 0.5 else "Poor"))
    metrics_table.add_row(
        "Sharpe Ratio",
        f"[{sharpe_color}]{sharpe:.3f}[/{sharpe_color}]",
        sharpe_interp
    )
    
    # CVaR / Tail Risk metrics (if available)
    cvar_info = portfolio_result.get("cvar_constraint", {})
    if cvar_info and isinstance(cvar_info, dict):
        # Add spacer
        metrics_table.add_row("", "", "")
        metrics_table.add_row("[bold]Tail Risk (CVaR @ 95%)[/bold]", "", "[bold]Anti-Lehman Insurance[/bold]")
        
        # CVaR (Expected Shortfall)
        cvar = cvar_info.get("cvar_constrained", float('nan'))
        if np.isfinite(cvar):
            cvar_pct = cvar * 100
            # Color: green if small loss, yellow if moderate, red if large
            if cvar > -0.10:  # Less than -10% loss
                cvar_color = "green"
                cvar_interp = "Low tail risk"
            elif cvar > -0.20:  # -10% to -20% loss
                cvar_color = "yellow"
                cvar_interp = "Moderate tail risk"
            else:  # Worse than -20%
                cvar_color = "red"
                cvar_interp = "High tail risk"
            
            metrics_table.add_row(
                "CVaR₉₅ (Expected Shortfall)",
                f"[{cvar_color}]{cvar:+.4f}[/{cvar_color}]",
                f"{cvar_interp} (≈ {cvar_pct:.2f}%)"
            )
        
        # VaR (Value at Risk)
        tail_metrics = cvar_info.get("tail_risk_metrics", {})
        var = tail_metrics.get("var", float('nan'))
        if np.isfinite(var):
            var_pct = var * 100
            metrics_table.add_row(
                "VaR₉₅ (5th percentile)",
                f"{var:+.4f}",
                f"≈ {var_pct:.2f}% threshold"
            )
        
        # Worst case scenario
        worst_case = tail_metrics.get("worst_case", float('nan'))
        if np.isfinite(worst_case):
            worst_pct = worst_case * 100
            metrics_table.add_row(
                "Worst Case (min return)",
                f"[indian_red1]{worst_case:+.4f}[/indian_red1]",
                f"≈ {worst_pct:.2f}% max loss"
            )
        
        # Constraint status
        constraint_active = cvar_info.get("constraint_active", False)
        scaling_factor = cvar_info.get("scaling_factor", 1.0)
        r_max = cvar_info.get("r_max", -0.20)
        
        if constraint_active:
            status_str = f"[yellow]ACTIVE[/yellow] (scaled by {scaling_factor:.2f}×)"
            interp = f"Reduced to meet {r_max*100:.0f}% max loss"
        else:
            status_str = "[#00d700]INACTIVE[/#00d700]"
            interp = f"Within {r_max*100:.0f}% limit"
        
        metrics_table.add_row(
            "CVaR Constraint",
            status_str,
            interp
        )
    
    console.print(metrics_table)
    
    # Correlation matrix table
    corr_table = Table(
        title="🔗 Asset Correlation Matrix",
        show_header=True,
        header_style="bold blue"
    )
    corr_table.add_column("", justify="left", style="bold")
    for name in asset_names:
        corr_table.add_column(name[:8], justify="right")
    
    for i, name in enumerate(asset_names):
        row_values = [name[:8]]
        for j in range(len(asset_names)):
            corr = correlation_matrix[i, j]
            if i == j:
                corr_str = "1.00"
            else:
                # Color code correlations (pleasant, non-neon colors)
                if abs(corr) > 0.7:
                    corr_str = f"[indian_red1]{corr:+.2f}[/indian_red1]"
                elif abs(corr) > 0.4:
                    corr_str = f"[yellow]{corr:+.2f}[/yellow]"
                else:
                    corr_str = f"[#00d700]{corr:+.2f}[/#00d700]"
            row_values.append(corr_str)
        corr_table.add_row(*row_values)
    
    console.print(corr_table)
    
    # Add explanatory caption
    console.print(
        "\n[dim]💡 Expected Utility: Optimizes EU = p×E[gain] - (1-p)×E[loss] from posterior predictive samples.[/dim]"
    )
    console.print(
        "[dim]   Position size = EU / max(E[loss], ε), with covariance adjustment for portfolio allocation.[/dim]"
    )
    console.print(
        "[dim]   Diversification ratio < 1 indicates correlation benefits captured.[/dim]\n"
    )


# Column descriptions for exports
DETAILED_COLUMN_DESCRIPTIONS = {
    "horizon_trading_days": "Number of trading days in the forecast horizon.",
    "edge_z_risk_adjusted": "Risk-adjusted edge (z-score) combining drift/vol with momentum/trend filters.",
    "prob_up": "Estimated probability the horizon return is positive (Student-t by default).",
    "expected_log_return": "Expected cumulative log return over the horizon from the daily drift estimate.",
    "ci_low_log": "Lower bound of the two-sided confidence interval for log return.",
    "ci_high_log": "Upper bound of the two-sided confidence interval for log return.",
    "profit_pln_on_1m_pln": "Expected profit in Polish zloty (PLN) when investing 1,000,000 PLN.",
    "profit_ci_low_pln": "Lower confidence bound for profit (PLN) on 1,000,000 PLN.",
    "profit_ci_high_pln": "Upper confidence bound for profit (PLN) on 1,000,000 PLN.",
    "signal": "Decision label based on prob_up: BUY (>=58%), HOLD (42–58%), SELL (<=42%).",
}

SIMPLIFIED_COLUMN_DESCRIPTIONS = {
    "timeframe": "Plain-English period (e.g., 1 day, 1 week, 3 months).",
    "chance_up": "Chance that the price goes up over this period (percent).",
    "recommendation": "BUY (expect price to rise), HOLD (unclear), or SELL (expect price to fall).",
    "why": "Short explanation combining trend, momentum, and volatility context.",
}


# =============================================================================
# NOTE: render_risk_temperature_summary is now imported from risk_temperature.py
# at the top of this file for backward compatibility.
# The canonical implementation lives in decision/risk_temperature.py.
# =============================================================================
