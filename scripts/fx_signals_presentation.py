#!/usr/bin/env python3
"""
fx_signals_presentation.py

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
        
        # Strong signals: ▲▲▼▼
        if label_upper.startswith("STRONG BUY"):
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
            signal_badge = "[indian_red1]↓ SELL[/]"
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
    """Render world-class compact signal heatmap table."""
    if not summary_rows:
        return

    # Sort: SELL first, then HOLD, then BUY, then STRONG BUY
    def signal_sort_key(row: Dict) -> int:
        nearest_label = row.get("nearest_label", "HOLD")
        label_upper = str(nearest_label).upper()
        if "SELL" in label_upper:
            return 0
        elif "HOLD" in label_upper:
            return 1
        elif "STRONG" in label_upper and "BUY" in label_upper:
            return 3
        elif "BUY" in label_upper:
            return 2
        return 1

    sorted_rows = sorted(summary_rows, key=signal_sort_key)

    # Compact asset column width
    import re
    def _plain_len(text: str) -> int:
        if not isinstance(text, str):
            return 0
        return len(re.sub(r"\[/?[^\]]+\]", "", text))
    if asset_col_width is None:
        longest_asset = max((_plain_len(r.get("asset_label", "")) for r in sorted_rows), default=0)
        asset_col_width = max(40, min(52, longest_asset + 4))

    if console is None:
        console = Console(force_terminal=True, width=200)
    
    # Clean table with vertical columns
    table = Table(
        title=title_override,
        show_header=True,
        header_style="bold white",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1),
        collapse_padding=False,
        row_styles=["", "on grey7"],  # Alternating row colors
    )
    # Asset column - generous width for full names
    table.add_column("Asset", justify="left", style="white", width=asset_col_width, no_wrap=True, overflow="ellipsis")
    # Exhaustion columns
    table.add_column("↑", justify="right", width=3, style="indian_red1")  # Overbought
    table.add_column("↓", justify="right", width=3, style="bright_green")  # Oversold
    
    # Horizon columns - slightly wider for readability
    horizon_labels = {1: "1d", 3: "3d", 7: "1w", 21: "1m", 63: "3m", 126: "6m", 252: "12m"}
    for horizon in horizons:
        label = horizon_labels.get(horizon, f"{horizon}d")
        table.add_column(label, justify="center", width=11, no_wrap=True)

    for row in sorted_rows:
        asset_label = row.get("asset_label", "Unknown")
        horizon_signals = row.get("horizon_signals", {})
        
        # Compute max UE↑ and UE↓ across all horizons for this asset
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
        
        # Format UE↑ as percentage (0-100%) with color coding
        # Show dot if value is not meaningful (< 5%)
        ue_up_pct = int(max_ue_up * 100)
        if ue_up_pct < 5:
            ue_up_display = "[dim]·[/dim]"
        elif max_ue_up >= 0.6:
            ue_up_display = f"[indian_red1]{ue_up_pct}[/indian_red1]"
        elif max_ue_up >= 0.3:
            ue_up_display = f"[yellow]{ue_up_pct}[/yellow]"
        else:
            ue_up_display = f"[dim]{ue_up_pct}[/dim]"

        # Format UE↓ as percentage (0-100%) with color coding
        # Show dot if value is not meaningful (< 5%)
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
        
        table.add_row(asset_label, ue_up_display, ue_down_display, *cells)

    console.print(table)
    console.print()


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
    
    exhaust_legend.add_row(
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
    console.print(Rule(style="dim"))
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
# TUNING OUTPUT PRESENTATION
# =============================================================================
# World-class UX for Kalman MLE tuning pipeline output
# Apple Design Principles: Clarity, Deference, Depth
# =============================================================================

# Regime labels for display (imported from tune_q_mle or defined here for standalone use)
TUNING_REGIME_LABELS = {
    0: "LOW_VOL_TREND",
    1: "HIGH_VOL_TREND",
    2: "LOW_VOL_RANGE",
    3: "HIGH_VOL_RANGE",
    4: "CRISIS_JUMP",
}

# Regime color mapping for visual hierarchy
REGIME_COLORS = {
    "LOW_VOL_TREND": "cyan",
    "HIGH_VOL_TREND": "yellow",
    "LOW_VOL_RANGE": "green",
    "HIGH_VOL_RANGE": "orange1",
    "CRISIS_JUMP": "red",
}


def create_tuning_console() -> Console:
    """Create a console with optimal settings for tuning output."""
    return Console(force_terminal=True, color_system="truecolor", width=140)


def _human_number(n: int) -> str:
    """Format number with K/M suffix."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def render_tuning_header(
    prior_mean: float,
    prior_lambda: float,
    lambda_regime: float,
    console: Console = None
) -> None:
    """Render extraordinary Apple-quality header for tuning pipeline.
    
    Apple Design Principles Applied:
    1. Clarity - Clear visual hierarchy, no clutter
    2. Deference - Content first, chrome minimal  
    3. Depth - Layered information density
    """
    if console is None:
        console = create_tuning_console()
    
    from rich.align import Align
    from datetime import datetime
    import multiprocessing
    
    # Clear for immersive experience
    console.clear()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # HERO TITLE - Cinematic, minimal
    # ═══════════════════════════════════════════════════════════════════════════════
    console.print()
    console.print()
    
    title = Text()
    title.append("◆", style="bold bright_cyan")
    title.append("  K A L M A N   T U N E R", style="bold bright_white")
    console.print(Align.center(title))
    
    subtitle = Text("Hierarchical Regime-Conditional Maximum Likelihood", style="dim")
    console.print(Align.center(subtitle))
    
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # CONTEXT BAR - Time, cores, date - ultra minimal
    # ═══════════════════════════════════════════════════════════════════════════════
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
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # PRIORS CARD - Clean, scannable
    # ═══════════════════════════════════════════════════════════════════════════════
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
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # MODEL CHIPS - Elegant badges
    # ═══════════════════════════════════════════════════════════════════════════════
    chips = Text()
    chips.append("  ○ ", style="green")
    chips.append("Gaussian", style="green")
    chips.append("    ○ ", style="cyan")
    chips.append("φ-Gaussian", style="cyan")
    chips.append("    ○ ", style="magenta")
    chips.append("φ-Student-t", style="magenta")
    chips.append(" ", style="dim")
    chips.append("(ν ∈ {4, 6, 8, 12, 20})", style="dim")
    console.print(Align.center(chips))
    
    console.print()
    console.print()


def render_tuning_progress_start(
    n_assets: int,
    n_workers: int,
    n_cached: int,
    cache_size: int,
    cache_path: str,
    console: Console = None
) -> None:
    """Render extraordinary Apple-quality processing phase.
    
    Design: Elegant integrated info card with visual hierarchy
    """
    if console is None:
        console = create_tuning_console()
    
    from rich.align import Align
    from rich.rule import Rule
    from datetime import datetime
    
    console.print()
    console.print(Rule(style="dim", characters="─"))
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # ESTIMATION PHASE - Clean, integrated
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Phase title
    title = Text()
    title.append("▸ ", style="bright_yellow")
    title.append("ESTIMATION", style="bold white")
    console.print(title)
    console.print()
    
    # Stats in elegant horizontal layout
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


def render_cache_status(
    cache_size: int,
    cache_path: str,
    console: Console = None
) -> None:
    """Render elegant cache status - one line, clean."""
    if console is None:
        console = create_tuning_console()
    
    filename = cache_path.split('/')[-1]
    console.print(f"  [dim]Cache:[/dim] [white]{cache_size:,}[/white] [dim]entries in[/dim] [white]{filename}[/white]")


def render_cache_update(
    cache_path: str,
    console: Console = None
) -> None:
    """Render cache update confirmation - subtle."""
    if console is None:
        console = create_tuning_console()
    
    console.print(f"  [green]✓[/green] [dim]Saved[/dim]")


def render_asset_progress(
    asset: str,
    index: int,
    total: int,
    status: str,
    details: Optional[str] = None,
    console: Console = None
) -> None:
    """Render single asset progress - compact."""
    if console is None:
        console = create_tuning_console()
    
    icons = {
        'success': '[green]✓[/green]',
        'cached': '[blue]○[/blue]',
        'failed': '[red]✗[/red]',
        'warning': '[yellow]![/yellow]',
    }
    
    icon = icons.get(status, '·')
    detail_str = f" [dim]{details}[/dim]" if details else ""
    console.print(f"    {icon} [white]{asset}[/white]{detail_str}")


def _get_status(fit_count: int, shrunk_count: int) -> str:
    """Get plain text status for regime row."""
    if fit_count == 0:
        return "—"
    elif shrunk_count > 0:
        pct = shrunk_count / fit_count * 100 if fit_count > 0 else 0
        return f"{pct:.0f}%"
    else:
        return "✓"


def render_tuning_summary(
    total_assets: int,
    new_estimates: int,
    reused_cached: int,
    failed: int,
    calibration_warnings: int,
    gaussian_count: int,
    student_t_count: int,
    regime_tuning_count: int,
    lambda_regime: float,
    regime_fit_counts: Dict[int, int],
    regime_shrunk_counts: Dict[int, int],
    collapse_warnings: int,
    cache_path: str,
    regime_model_breakdown: Optional[Dict[int, Dict[str, int]]] = None,
    console: Console = None
) -> None:
    """Render extraordinary Apple-quality tuning summary.
    
    Design: Clean cards, clear hierarchy, breathing room
    """
    if console is None:
        console = create_tuning_console()
    
    from rich.align import Align
    from rich.rule import Rule
    
    console.print()
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # COMPLETION HEADER - Elegant centered title
    # ═══════════════════════════════════════════════════════════════════════════════
    header_text = Text(justify="center")
    header_text.append("\n", style="")
    header_text.append("✓ ", style="bold bright_green")
    header_text.append("TUNING COMPLETE", style="bold bright_white")
    header_text.append("\n", style="")
    
    header_panel = Panel(
        Align.center(header_text),
        box=box.ROUNDED,
        border_style="bright_green",
        padding=(0, 4),
        width=40,
    )
    console.print(Align.center(header_panel))
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # METRICS ROW - Clean cards with clear labels
    # ═══════════════════════════════════════════════════════════════════════════════
    console.print()
    
    metrics_table = Table(
        show_header=False,
        box=None,
        padding=(0, 4),
        expand=False,
    )
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
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # MODEL SELECTION - Visual comparison with elegant bars
    # ═══════════════════════════════════════════════════════════════════════════════
    total_models = gaussian_count + student_t_count
    if total_models > 0:
        console.print(Rule(style="dim"))
        console.print()
        
        section = Text()
        section.append("  📈  ", style="bold bright_cyan")
        section.append("MODEL SELECTION", style="bold bright_white")
        console.print(section)
        console.print()
        
        gauss_pct = gaussian_count / total_models * 100
        student_pct = student_t_count / total_models * 100
        
        # Visual bars
        bar_width = 35
        gauss_filled = int(gauss_pct / 100 * bar_width)
        student_filled = int(student_pct / 100 * bar_width)
        
        # Gaussian row
        gauss_row = Text()
        gauss_row.append("    ○ ", style="green")
        gauss_row.append("Gaussian      ", style="green")
        gauss_row.append("█" * gauss_filled, style="green")
        gauss_row.append("░" * (bar_width - gauss_filled), style="dim")
        gauss_row.append(f"  {gaussian_count:>4}", style="bold white")
        gauss_row.append(f"  ({gauss_pct:>4.1f}%)", style="dim")
        console.print(gauss_row)
        
        # Student-t row
        student_row = Text()
        student_row.append("    ● ", style="magenta")
        student_row.append("Student-t     ", style="magenta")
        student_row.append("█" * student_filled, style="magenta")
        student_row.append("░" * (bar_width - student_filled), style="dim")
        student_row.append(f"  {student_t_count:>4}", style="bold white")
        student_row.append(f"  ({student_pct:>4.1f}%)", style="dim")
        console.print(student_row)
        
        console.print()
        console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # REGIME TABLE - With model breakdown in elegant table
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
    
    # Collect all model types across all regimes
    all_model_types = set()
    if regime_model_breakdown:
        for r_breakdown in regime_model_breakdown.values():
            all_model_types.update(r_breakdown.keys())
    
    # Sort model types
    def model_sort_key(m):
        if m == "Gaussian":
            return (0, 0)
        elif m == "φ-Gaussian":
            return (1, 0)
        else:
            import re
            nu_match = re.search(r'ν=(\d+)', m)
            nu = int(nu_match.group(1)) if nu_match else 0
            return (2, nu)
    
    sorted_models = sorted(all_model_types, key=model_sort_key)
    
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
    table.add_column("Fits", justify="right", width=6)
    table.add_column("Distribution", width=25)
    
    # Add columns for each model type
    for model in sorted_models:
        if model == "Gaussian":
            table.add_column("G", justify="right", width=4, style="green")
        elif model == "φ-Gaussian":
            table.add_column("φ-G", justify="right", width=4, style="cyan")
        else:
            import re
            nu_match = re.search(r'ν=(\d+)', model)
            nu = nu_match.group(1) if nu_match else "?"
            table.add_column(f"t{nu}", justify="right", width=4, style="magenta")
    
    for i, (name, short, color, icon) in enumerate(zip(regime_names, regime_short, regime_colors_list, regime_icons)):
        fit_count = regime_fit_counts.get(i, 0)
        
        # Create visual bar
        if fit_count == 0:
            bar = "[dim]" + "─" * 20 + "[/]"
        else:
            filled = int(fit_count / max_fits * 20) if max_fits > 0 else 0
            bar = f"[{color}]{'━' * filled}[/{color}][dim]{'─' * (20 - filled)}[/]"
        
        # Build row
        row = [
            f"[{color}]{icon} {short}[/{color}]",
            f"[bold]{fit_count}[/]" if fit_count > 0 else "[dim]0[/]",
            bar,
        ]
        
        # Add counts for each model type
        if regime_model_breakdown:
            r_breakdown = regime_model_breakdown.get(i, {})
            for model in sorted_models:
                count = r_breakdown.get(model, 0)
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


def render_parameter_table(
    cache: Dict[str, Dict],
    console: Console = None
) -> None:
    """Render beautiful parameter table - clean, scannable, Apple-quality design."""
    if console is None:
        console = create_tuning_console()
    
    if not cache:
        return
    
    import numpy as np
    
    def _model_label(data: dict) -> str:
        if 'global' in data:
            data = data['global']
        noise_model = data.get('noise_model', 'gaussian')
        if noise_model and noise_model.startswith('phi_student_t_nu_'):
            return 'Student-t'
        if noise_model == 'kalman_phi_gaussian' or data.get('phi') is not None:
            return 'φ-Gaussian'
        return 'Gaussian'
    
    def _get_q_for_sort(data):
        if 'global' in data:
            return data['global'].get('q', 0)
        return data.get('q', 0)
    
    # Group by model
    groups: Dict[str, List] = {}
    for asset, data in cache.items():
        model = _model_label(data)
        if model not in groups:
            groups[model] = []
        groups[model].append((asset, data))
    
    # Sort each group
    for model in groups:
        groups[model].sort(key=lambda x: -_get_q_for_sort(x[1]))
    
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    
    section = Text()
    section.append("  📊  ", style="bold bright_cyan")
    section.append("TUNED PARAMETERS", style="bold bright_white")
    console.print(section)
    console.print()
    
    model_colors = {'Gaussian': 'green', 'φ-Gaussian': 'cyan', 'Student-t': 'magenta'}
    model_icons = {'Gaussian': '○', 'φ-Gaussian': '◐', 'Student-t': '●'}
    
    for model_name in ['Gaussian', 'φ-Gaussian', 'Student-t']:
        if model_name not in groups:
            continue
        
        assets = groups[model_name]
        color = model_colors.get(model_name, 'white')
        icon = model_icons.get(model_name, '·')
        
        # Model header with count
        console.print()
        header = Text()
        header.append(f"  {icon} ", style=f"bold {color}")
        header.append(f"{model_name}", style=f"bold {color}")
        header.append(f"  ({len(assets)} assets)", style="dim")
        console.print(header)
        console.print()
        
        # Create elegant table with proper borders
        table = Table(
            show_header=True,
            header_style="bold white",
            border_style="dim",
            box=box.ROUNDED,
            padding=(0, 1),
            row_styles=["", "on grey7"],
        )
        table.add_column("Asset", style="bold", width=14, no_wrap=True)
        table.add_column("log₁₀(q)", justify="right", width=10)
        table.add_column("c", justify="right", width=8)
        table.add_column("φ", justify="right", width=8)
        table.add_column("ν", justify="right", width=6)
        table.add_column("BIC", justify="right", width=10)
        table.add_column("PIT p", justify="right", width=8)
        
        for asset, raw_data in assets:
            if 'global' in raw_data:
                data = raw_data['global']
            else:
                data = raw_data
            
            q_val = data.get('q', float('nan'))
            c_val = data.get('c', 1.0)
            phi_val = data.get('phi')
            nu_val = data.get('nu')
            bic_val = data.get('bic', float('nan'))
            pit_p = data.get('pit_ks_pvalue', float('nan'))
            
            log10_q = np.log10(q_val) if q_val > 0 else float('nan')
            
            # Format values with proper styling
            q_str = f"{log10_q:.2f}" if np.isfinite(log10_q) else "[dim]—[/]"
            c_str = f"{c_val:.3f}"
            phi_str = f"{phi_val:+.3f}" if phi_val is not None else "[dim]—[/]"
            nu_str = f"{nu_val:.0f}" if nu_val is not None else "[dim]—[/]"
            bic_str = f"{bic_val:,.0f}" if np.isfinite(bic_val) else "[dim]—[/]"
            
            # Color PIT p-value based on calibration
            if np.isfinite(pit_p):
                if pit_p < 0.01:
                    pit_str = f"[bold indian_red1]{pit_p:.4f}[/]"
                elif pit_p < 0.05:
                    pit_str = f"[yellow]{pit_p:.4f}[/]"
                else:
                    pit_str = f"[bright_green]{pit_p:.4f}[/]"
            else:
                pit_str = "[dim]—[/]"
            
            table.add_row(asset, q_str, c_str, phi_str, nu_str, bic_str, pit_str)
        
        console.print(table)
    
    console.print()


def render_failed_assets(
    failure_reasons: Dict[str, str],
    console: Console = None
) -> None:
    """Render failed assets - clean, actionable, Apple-quality design."""
    if console is None:
        console = create_tuning_console()
    
    if not failure_reasons:
        return
    
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    
    section = Text()
    section.append("  ❌  ", style="bold indian_red1")
    section.append("FAILED ASSETS", style="bold indian_red1")
    section.append(f"  ({len(failure_reasons)})", style="dim")
    console.print(section)
    console.print()
    
    # Create table for failed assets
    table = Table(
        show_header=True,
        header_style="bold white",
        border_style="indian_red1",
        box=box.ROUNDED,
        padding=(0, 1),
        row_styles=["", "on grey7"],
    )
    table.add_column("Asset", style="bold indian_red1", width=15, no_wrap=True)
    table.add_column("Error", style="dim", width=70, no_wrap=True, overflow="ellipsis")
    
    for asset, reason in sorted(failure_reasons.items()):
        first_line = reason.split('\n')[0][:65] if reason else "Unknown error"
        table.add_row(asset, first_line)
    
    console.print(table)
    console.print()
    
    # Action hint
    hint = Text()
    hint.append("    → ", style="dim")
    hint.append("Re-run with ", style="dim")
    hint.append("make tune ARGS='--force --assets <TICKER>'", style="bold white")
    hint.append(" to retry", style="dim")
    console.print(hint)
    console.print()


def render_dry_run_preview(
    assets: List[str],
    max_display: int = 20,
    console: Console = None
) -> None:
    """Render dry run preview - clean, informative, Apple-quality design."""
    if console is None:
        console = create_tuning_console()
    
    from rich.align import Align
    
    console.print()
    console.print()
    
    # Warning panel
    warning_text = Text(justify="center")
    warning_text.append("\n", style="")
    warning_text.append("⚠️  DRY RUN MODE", style="bold bright_yellow")
    warning_text.append("\n", style="")
    warning_text.append("No changes will be made to cache", style="dim")
    warning_text.append("\n", style="")
    
    warning_panel = Panel(
        Align.center(warning_text),
        box=box.ROUNDED,
        border_style="yellow",
        padding=(0, 4),
        width=45,
    )
    console.print(Align.center(warning_panel))
    console.print()
    
    # Assets list header
    header = Text()
    header.append("  📋  ", style="bold bright_cyan")
    header.append(f"Would process ", style="white")
    header.append(f"{len(assets)}", style="bold bright_cyan")
    header.append(" assets:", style="white")
    console.print(header)
    console.print()
    
    # Create table for assets
    table = Table(
        show_header=True,
        header_style="bold white",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1),
        row_styles=["", "on grey7"],
    )
    table.add_column("#", justify="right", width=4, style="dim")
    table.add_column("Asset", style="bold", width=15)
    table.add_column("Status", width=15)
    
    for i, asset in enumerate(assets[:max_display], 1):
        table.add_row(
            f"{i}",
            asset,
            "[dim]pending[/]"
        )
    
    if len(assets) > max_display:
        table.add_row(
            "[dim]...[/]",
            f"[dim]+ {len(assets) - max_display} more[/]",
            ""
        )
    
    console.print(table)
    console.print()
    
    # Hint
    hint = Text()
    hint.append("    → ", style="dim")
    hint.append("Remove ", style="dim")
    hint.append("--dry-run", style="bold white")
    hint.append(" to execute", style="dim")
    console.print(hint)
    console.print()


def render_end_of_run_summary(
    processed_assets: Dict[str, Dict],
    regime_distributions: Dict[str, Dict[int, int]],
    model_comparisons: Dict[str, Dict],
    failure_reasons: Dict[str, str],
    processing_log: List[str],
    console: Console = None,
    cache: Dict = None,
) -> None:
    """Render end-of-run summary - optional verbose details."""
    if console is None:
        console = create_tuning_console()
    
    # This is called for verbose output - keep it minimal unless user requests details
    if failure_reasons:
        render_failed_assets(failure_reasons, console=console)
    
    # Render calibration report if cache is provided
    if cache:
        render_calibration_report(cache, failure_reasons, console=console)


def render_calibration_report(
    cache: Dict,
    failure_reasons: Dict[str, str],
    console: Console = None
) -> None:
    """Render Apple-quality calibration report showing assets with issues.
    
    Shows:
    - PIT p-value < 0.05 (model predictions not well-calibrated)
    - High kurtosis (fat tails not captured)
    - Failed tuning
    - Regime collapse warnings
    
    Also saves issues to JSON file: scripts/quant/cache/calibration/calibration_failures.json
    """
    import numpy as np
    import json
    import os
    from datetime import datetime
    
    if console is None:
        console = create_tuning_console()
    
    # Collect calibration issues
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
            'details': reason[:200] if reason else ''
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
        
        # Check for mixture model usage
        mixture_selected = data.get('mixture_selected', False)
        mixture_model = data.get('mixture_model')
        
        collapse_warning = raw_data.get('hierarchical_tuning', {}).get('collapse_warning', False)
        
        has_issue = False
        issue_type = []
        severity = 'ok'
        
        # If mixture was selected and improved calibration, skip this asset
        if mixture_selected and mixture_model and not calibration_warning:
            continue
        
        if calibration_warning or (pit_p is not None and pit_p < 0.05):
            has_issue = True
            # Note if mixture was attempted but didn't help
            if mixture_selected:
                issue_type.append('PIT < 0.05 (mix)')
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
        
        if collapse_warning:
            has_issue = True
            issue_type.append('Regime Collapse')
        
        if has_issue:
            if mixture_selected and mixture_model:
                # Show mixture model info
                sigma_ratio = mixture_model.get('sigma_ratio', 0)
                model_str = f"Mix(σ={sigma_ratio:.1f})"
            elif 'student_t' in noise_model:
                model_str = f"φ-T(ν={int(nu_val)})" if nu_val else "Student-t"
            elif 'phi' in noise_model:
                model_str = "φ-Gaussian"
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
                'mixture_selected': mixture_selected,
            })
    
    # Sort by severity (critical first), then by PIT p-value
    severity_order = {'critical': 0, 'warning': 1, 'ok': 2}
    issues.sort(key=lambda x: (severity_order.get(x['severity'], 2), x.get('pit_p') or 1.0))
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # SAVE CALIBRATION ISSUES TO JSON FILE
    # ═══════════════════════════════════════════════════════════════════════════════
    calibration_dir = os.path.join(os.path.dirname(__file__), 'quant', 'cache', 'calibration')
    os.makedirs(calibration_dir, exist_ok=True)
    calibration_file = os.path.join(calibration_dir, 'calibration_failures.json')
    
    # Prepare JSON-serializable data
    total_assets = len(cache) if cache else 0
    critical_count = sum(1 for i in issues if i['severity'] == 'critical')
    warning_count = sum(1 for i in issues if i['severity'] == 'warning')
    failed_count = sum(1 for i in issues if i['issue_type'] == 'FAILED')
    
    calibration_report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_assets': total_assets,
            'total_issues': len(issues),
            'critical': critical_count,
            'warnings': warning_count,
            'failed': failed_count,
            'passed': total_assets - len(issues) if total_assets > 0 else 0,
        },
        'issues': [
            {
                'asset': issue['asset'],
                'issue_type': issue['issue_type'],
                'severity': issue['severity'],
                'pit_ks_pvalue': issue['pit_p'],
                'ks_statistic': issue['ks_stat'],
                'kurtosis': issue['kurtosis'],
                'model': issue['model'],
                'q': issue['q'],
                'phi': issue['phi'],
                'nu': issue['nu'],
                'details': issue['details'],
                'mixture_attempted': issue.get('mixture_selected', False),
            }
            for issue in issues
        ],
        'thresholds': {
            'pit_warning': 0.05,
            'pit_critical': 0.01,
            'kurtosis_warning': 6.0,
            'kurtosis_critical': 10.0,
        },
        'mixture_model': {
            'enabled': True,
            'description': 'K=2 symmetric φ-t mixture for calibration improvement',
            'sigma_ratio_min': 1.5,
            'weight_bounds': [0.1, 0.9],
        }
    }
    
    try:
        with open(calibration_file, 'w') as f:
            json.dump(calibration_report, f, indent=2, default=str)
    except Exception as e:
        # Don't fail the report if we can't save the file
        pass
    
    # SECTION HEADER - Always show
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    
    section_header = Text()
    section_header.append("  📊  ", style="bold bright_cyan")
    section_header.append("CALIBRATION REPORT", style="bold bright_white")
    console.print(section_header)
    console.print()
    
    # Show success or issues
    if not issues:
        console.print()
        success_text = Text()
        success_text.append("  ✓ ", style="bold bright_green")
        success_text.append("All ", style="white")
        success_text.append(f"{total_assets}", style="bold bright_cyan")
        success_text.append(" assets passed calibration checks", style="white")
        console.print(success_text)
        console.print()
        
        stats_text = Text()
        stats_text.append("    PIT p-value ≥ 0.05 for all models  ·  ", style="dim")
        stats_text.append("No regime collapse detected", style="dim")
        console.print(stats_text)
        console.print()
        return
    
    # ISSUES HEADER
    issues_header = Text()
    issues_header.append("  ⚠️  ", style="bold yellow")
    issues_header.append(f"{len(issues)} assets with calibration issues", style="bold yellow")
    console.print(issues_header)
    console.print()
    
    # SUMMARY STATS (use counts computed earlier)
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
    
    # ISSUES TABLE
    table = Table(
        show_header=True,
        header_style="bold white",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1),
        row_styles=["", "on grey7"],
    )
    
    table.add_column("Asset", justify="left", width=30, no_wrap=True)
    table.add_column("Issue", justify="left", width=18)
    table.add_column("PIT p", justify="right", width=8)
    table.add_column("KS", justify="right", width=6)
    table.add_column("Kurt", justify="right", width=6)
    table.add_column("Model", justify="left", width=12)
    table.add_column("log₁₀(q)", justify="right", width=9)
    table.add_column("φ", justify="right", width=6)
    table.add_column("Details", justify="left", width=25, no_wrap=True)
    
    for issue in issues:
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
                pit_styled = f"[dim]{pit_str}[/]"
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
            f"[dim]{issue['details']}[/]",
        )
    
    console.print(table)
    console.print()
    
    # LEGEND
    legend = Text()
    legend.append("    ", style="")
    legend.append("PIT p < 0.05", style="yellow")
    legend.append(" = model may be miscalibrated   ·   ", style="dim")
    legend.append("Kurt > 6", style="yellow")
    legend.append(" = heavy tails not fully captured", style="dim")
    
    console.print(legend)
    console.print()
    
    # Action recommendation
    if critical_count > 0:
        action = Text()
        action.append("    → ", style="dim")
        action.append("Consider re-tuning critical assets with ", style="dim")
        action.append("make tune ARGS='--force --assets <TICKER>'", style="bold white")
        console.print(action)
        console.print()
    
    # Show where file was saved
    file_info = Text()
    file_info.append("    💾 ", style="dim")
    file_info.append("Saved to ", style="dim")
    file_info.append("scripts/quant/cache/calibration/calibration_failures.json", style="dim italic")
    console.print(file_info)
    console.print()


class TuningProgressTracker:
    """Extraordinary Apple-quality progress tracker.
    
    Design: Clean separation between progress bar and completion log
    """

    def __init__(self, total_assets: int, console: Console = None):
        self.total = total_assets
        self.console = console or create_tuning_console()
        self.current = 0
        self.successes = 0
        self.failures = 0
        self.completed = []  # Store completed assets for final display
        
        # Show progress bar first
        self.progress = Progress(
            SpinnerColumn(spinner_name="dots", style="bright_yellow"),
            BarColumn(
                bar_width=40, 
                complete_style="bright_green", 
                finished_style="bright_green",
            ),
            TaskProgressColumn(),
            TextColumn("[dim]·[/dim]"),
            MofNCompleteColumn(),
            TextColumn("[dim]·[/dim]"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,  # Progress bar will be replaced
        )
        self.progress.start()
        self.task = self.progress.add_task("", total=total_assets)

    def _model_badge(self, details: str) -> Text:
        """Create elegant model badge."""
        badge = Text()
        if not details:
            return badge
        
        details_lower = details.lower()
        
        if 'student' in details_lower:
            import re
            nu_match = re.search(r'ν=(\d+)', details)
            badge.append("φ-t", style="magenta")
            if nu_match:
                badge.append(f"({nu_match.group(1)})", style="dim magenta")
        elif 'φ' in details or 'phi' in details_lower:
            badge.append("φ-G", style="cyan")
        else:
            badge.append("G", style="green")
        
        return badge

    def _q_value(self, details: str) -> Text:
        """Extract and format q value."""
        result = Text()
        if not details:
            return result
        
        import re
        q_match = re.search(r'q=([0-9.e+-]+)', details)
        if q_match:
            try:
                q_val = float(q_match.group(1))
                log_q = np.log10(q_val) if q_val > 0 else 0
                result.append(f"{log_q:+.1f}", style="dim")
            except:
                pass
        return result

    def update(self, asset: str, status: str, details: Optional[str] = None):
        """Update progress - store completions for later display."""
        self.current += 1
        
        if status == 'success':
            self.successes += 1
            self.completed.append((asset, details, 'success'))
        elif status == 'failed':
            self.failures += 1
            error_msg = details.split('\n')[0][:35] if details else "Error"
            self.completed.append((asset, error_msg, 'failed'))
        
        self.progress.update(self.task, advance=1)

    def _get_model_info(self, details: str) -> dict:
        """Extract all parameters from details string.
        
        Details format: model|q=X|c=X|φ=X|ν=X|bic=X
        """
        import re
        
        info = {
            'model': '—',
            'q': '—',
            'c': '—',
            'phi': '—',
            'nu': '—',
            'bic': '—',
        }
        
        if not details:
            return info
        
        details_lower = details.lower()
        
        # Determine model type
        if 'student' in details_lower:
            info['model'] = 'φ-t'
        elif 'φ-gaussian' in details_lower or 'phi' in details_lower:
            info['model'] = 'φ-G'
        else:
            info['model'] = 'G'
        
        # Extract q value
        q_match = re.search(r'q=([0-9.e+-]+)', details)
        if q_match:
            try:
                q_val = float(q_match.group(1))
                import math
                log_q = math.log10(q_val) if q_val > 0 else 0
                info['q'] = f"{log_q:.2f}"
            except:
                pass
        
        # Extract c value
        c_match = re.search(r'c=([0-9.]+)', details)
        if c_match:
            info['c'] = c_match.group(1)
        
        # Extract φ value
        phi_match = re.search(r'φ=([+-]?[0-9.]+)', details)
        if phi_match:
            info['phi'] = phi_match.group(1)
        
        # Extract ν value
        nu_match = re.search(r'ν=(\d+)', details)
        if nu_match:
            info['nu'] = nu_match.group(1)
        
        # Extract BIC value
        bic_match = re.search(r'bic=([+-]?[0-9.]+)', details)
        if bic_match:
            info['bic'] = bic_match.group(1)
        
        return info

    def finish(self):
        """Complete with elegant aligned summary showing all parameters."""
        self.progress.stop()
        
        # Now show completion log in its own section
        self.console.print()
        
        section = Text()
        section.append("▸ ", style="bright_green")
        section.append("COMPLETED", style="bold white")
        self.console.print(section)
        self.console.print()
        
        # Create a comprehensive parameter table
        table = Table(
            show_header=True,
            header_style="dim",
            box=None,
            padding=(0, 2),
            collapse_padding=True,
        )
        table.add_column("", width=2)  # Icon
        table.add_column("Asset", style="bold white", width=10)
        table.add_column("Model", width=6)
        table.add_column("log₁₀(q)", justify="right", width=8)
        table.add_column("c", justify="right", width=6)
        table.add_column("φ", justify="right", width=6)
        table.add_column("ν", justify="right", width=4)
        table.add_column("BIC", justify="right", width=8)
        
        for asset, details, status in self.completed:
            if status == 'success':
                info = self._get_model_info(details)
                
                # Color the model badge
                model = info['model']
                if model == 'φ-t':
                    model_styled = f"[magenta]{model}[/magenta]"
                elif model == 'φ-G':
                    model_styled = f"[cyan]{model}[/cyan]"
                else:
                    model_styled = f"[green]{model}[/green]"
                
                # Format phi with color
                phi_str = info['phi']
                if phi_str != '—':
                    phi_str = f"[white]{phi_str}[/white]"
                else:
                    phi_str = f"[dim]{phi_str}[/dim]"
                
                # Format nu
                nu_str = info['nu']
                if nu_str == '—':
                    nu_str = f"[dim]{nu_str}[/dim]"
                
                table.add_row(
                    "[green]✓[/green]",
                    asset,
                    model_styled,
                    info['q'],
                    info['c'],
                    phi_str,
                    nu_str,
                    f"[dim]{info['bic']}[/dim]"
                )
            else:
                table.add_row(
                    "[red]✗[/red]",
                    asset,
                    f"[dim]Error[/dim]",
                    "",
                    "",
                    "",
                    "",
                    f"[dim red]{details[:15]}[/dim red]"
                )
        
        self.console.print(Padding(table, (0, 0, 0, 4)))
        self.console.print()
        
        # Summary line
        summary = Text()
        summary.append("    ")
        summary.append(f"{self.successes}", style="bold green")
        summary.append(" estimated", style="dim")
        if self.failures > 0:
            summary.append("  ·  ", style="dim")
            summary.append(f"{self.failures}", style="bold red")
            summary.append(" failed", style="dim")
        self.console.print(summary)
        self.console.print()