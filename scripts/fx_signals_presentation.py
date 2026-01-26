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
    """Format signal with ultra-compact, high-visibility styling.
    
    World-class UX: Clean, scannable, color-coded signals.
    Visual hierarchy (strongest to weakest):
      - ▲▲▼▼ Double filled triangles = Strong signals (STRONG BUY/SELL)
      - △▽ Empty triangles = Notable HOLD moves (significant return but no signal)
      - ↑↓ Arrows = Regular signals (BUY/SELL)
      - No symbol = Small moves
    Color coding: Green=positive, Red=negative, Light Blue=neutral small HOLD
    """
    pct_return = (profit_pln / notional_pln * 100) if notional_pln > 0 else 0.0
    
    # Ultra-compact profit display
    abs_profit = abs(profit_pln)
    if abs_profit >= 1_000_000:
        profit_compact = f"{profit_pln/1_000_000:+.0f}M"
    elif abs_profit >= 1_000:
        profit_compact = f"{profit_pln/1_000:+.0f}k"
    else:
        profit_compact = f"{profit_pln:+.0f}"
    
    if isinstance(signal_label, str):
        label_upper = signal_label.upper()
        
        # Strong signals: double filled triangles ▲▲▼▼ (pleasant muted tones)
        if label_upper.startswith("STRONG BUY"):
            return f"[bold #00d700]▲▲{pct_return:+.1f}% ({profit_compact})[/bold #00d700]"
        elif label_upper.startswith("STRONG SELL"):
            return f"[bold indian_red1]▼▼{pct_return:+.1f}% ({profit_compact})[/bold indian_red1]"
        # Regular BUY/SELL: arrows ↑↓ (natural, pleasant colors)
        elif "SELL" in label_upper:
            return f"[indian_red1]↓{pct_return:+.1f}% ({profit_compact})[/indian_red1]"
        elif "BUY" in label_upper:
            return f"[#00d700]↑{pct_return:+.1f}% ({profit_compact})[/#00d700]"
        else:
            # HOLD: Notable moves use pleasant colors based on direction
            # Small moves use grey (neutral)
            if pct_return > 10.0:
                return f"[bold #00d700]△{pct_return:+.1f}% ({profit_compact})[/bold #00d700]"
            elif pct_return < -10.0:
                return f"[bold indian_red1]▽{pct_return:+.1f}% ({profit_compact})[/bold indian_red1]"
            elif pct_return > 3.0:
                return f"[#00d700]△{pct_return:+.1f}% ({profit_compact})[/#00d700]"
            elif pct_return < -3.0:
                return f"[indian_red1]▽{pct_return:+.1f}% ({profit_compact})[/indian_red1]"
            elif pct_return > 1.0:
                return f"[#a8a8a8]{pct_return:+.1f}% ({profit_compact})[/#a8a8a8]"
            elif pct_return < -1.0:
                return f"[#a8a8a8]{pct_return:+.1f}% ({profit_compact})[/#a8a8a8]"
            else:
                return f"[#8a8a8a]{pct_return:+.1f}% ({profit_compact})[/#8a8a8a]"
    
    return f"{pct_return:+.1f}% ({profit_compact})"


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
    """Render world-class detailed signal analysis table with risk metrics.
    
    Args:
        asset_symbol: Trading symbol
        title: Full descriptive title
        signals: List of Signal dataclass instances
        price_series: Historical price series
        confidence_level: Two-sided confidence level (e.g., 0.68)
        used_student_t_mapping: Whether Student-t CDF was used for probabilities
        show_caption: Whether to show detailed column explanations
    """
    console = Console(force_terminal=True)
    last_close = convert_to_float(price_series.iloc[-1])
    last_date = price_series.index[-1].date()

    # Create beautiful table with proper styling
    table = Table(
        show_header=True,
        header_style="bold white",
        border_style="cyan",
        padding=(0, 1),
        expand=False,
    )
    
    # Column headers with better names
    table.add_column("Horizon", justify="center", style="bold", width=8)
    table.add_column("Edge", justify="right", width=7)
    table.add_column("Prob ↑", justify="right", width=8)
    table.add_column("E[ret]", justify="right", width=9)
    table.add_column(f"CI {int(confidence_level*100)}%", justify="center", width=18)
    table.add_column("Strength", justify="right", width=9)
    table.add_column("Regime", justify="center", width=10)
    table.add_column("Profit (PLN)", justify="right", width=28)
    table.add_column("Signal", justify="center", width=12)

    # Regime color mapping
    regime_colors = {
        'crisis': 'red',
        'high_vol': 'yellow', 
        'normal': 'green',
        'low_vol': 'cyan',
        'trending': 'blue',
        'mean_revert': 'magenta',
    }

    for signal in signals:
        # Calculate percentage return for display
        notional = 1_000_000
        pct_return = signal.profit_pln / notional * 100
        pct_ci_low = signal.profit_ci_low_pln / notional * 100
        pct_ci_high = signal.profit_ci_high_pln / notional * 100
        
        # Format horizon with human-readable labels
        horizon_map = {1: "1d", 3: "3d", 7: "1w", 21: "1m", 63: "3m", 126: "6m", 252: "12m"}
        horizon_label = horizon_map.get(signal.horizon_days, f"{signal.horizon_days}d")
        
        # Color-code edge based on magnitude
        edge_val = signal.score
        if edge_val >= 1.0:
            edge_str = f"[bold #00d700]{edge_val:+.2f}[/bold #00d700]"
        elif edge_val >= 0.5:
            edge_str = f"[#00d700]{edge_val:+.2f}[/#00d700]"
        elif edge_val <= -1.0:
            edge_str = f"[bold indian_red1]{edge_val:+.2f}[/bold indian_red1]"
        elif edge_val <= -0.5:
            edge_str = f"[indian_red1]{edge_val:+.2f}[/indian_red1]"
        else:
            edge_str = f"{edge_val:+.2f}"
        
        # Color-code probability
        p_val = signal.p_up * 100
        if p_val >= 65:
            prob_str = f"[bold #00d700]{p_val:.1f}%[/bold #00d700]"
        elif p_val >= 55:
            prob_str = f"[#00d700]{p_val:.1f}%[/#00d700]"
        elif p_val <= 35:
            prob_str = f"[bold indian_red1]{p_val:.1f}%[/bold indian_red1]"
        elif p_val <= 45:
            prob_str = f"[indian_red1]{p_val:.1f}%[/indian_red1]"
        else:
            prob_str = f"{p_val:.1f}%"
        
        # Color-code expected return (pleasant, non-neon colors)
        exp_ret = signal.exp_ret
        if exp_ret >= 0.02:
            ret_str = f"[bold #00d700]{exp_ret:+.4f}[/bold #00d700]"
        elif exp_ret >= 0.005:
            ret_str = f"[#00d700]{exp_ret:+.4f}[/#00d700]"
        elif exp_ret <= -0.02:
            ret_str = f"[bold indian_red1]{exp_ret:+.4f}[/bold indian_red1]"
        elif exp_ret <= -0.005:
            ret_str = f"[indian_red1]{exp_ret:+.4f}[/indian_red1]"
        else:
            ret_str = f"[#9e9e9e]{exp_ret:+.4f}[/#9e9e9e]"
        
        # Format CI range
        ci_str = f"[{signal.ci_low:+.3f}, {signal.ci_high:+.3f}]"
        
        # Color-code position strength (pleasant, non-neon colors)
        strength = signal.position_strength
        if strength >= 0.7:
            strength_str = f"[bold #00d700]{strength:.2f}[/bold #00d700]"
        elif strength >= 0.3:
            strength_str = f"[#00d700]{strength:.2f}[/#00d700]"
        elif strength <= 0.1:
            strength_str = f"[#8a8a8a]{strength:.2f}[/#8a8a8a]"
        else:
            strength_str = f"[#9e9e9e]{strength:.2f}[/#9e9e9e]"
        
        # Color-code regime
        regime = signal.regime.lower() if signal.regime else "normal"
        regime_color = regime_colors.get(regime, 'white')
        regime_str = f"[{regime_color}]{signal.regime}[/{regime_color}]"
        
        # Format profit with compact display
        if abs(signal.profit_pln) >= 1_000_000:
            profit_compact = f"{signal.profit_pln/1_000_000:+.1f}M"
        elif abs(signal.profit_pln) >= 1_000:
            profit_compact = f"{signal.profit_pln/1_000:+.0f}k"
        else:
            profit_compact = f"{signal.profit_pln:+.0f}"
        
        # Color the profit based on value (pleasant, non-neon colors)
        if pct_return >= 5:
            profit_str = f"[bold #00d700]{profit_compact} ({pct_return:+.1f}%)[/bold #00d700]"
        elif pct_return >= 1:
            profit_str = f"[#00d700]{profit_compact} ({pct_return:+.1f}%)[/#00d700]"
        elif pct_return <= -5:
            profit_str = f"[bold indian_red1]{profit_compact} ({pct_return:+.1f}%)[/bold indian_red1]"
        elif pct_return <= -1:
            profit_str = f"[indian_red1]{profit_compact} ({pct_return:+.1f}%)[/indian_red1]"
        else:
            profit_str = f"[#9e9e9e]{profit_compact} ({pct_return:+.1f}%)[/#9e9e9e]"
        
        # Color-code signal label (pleasant, non-neon colors)
        label_upper = signal.label.upper() if signal.label else "HOLD"
        if "STRONG" in label_upper and "BUY" in label_upper:
            signal_str = f"[bold #00d700]▲▲ STRONG BUY[/bold #00d700]"
        elif "STRONG" in label_upper and "SELL" in label_upper:
            signal_str = f"[bold indian_red1]▼▼ STRONG SELL[/bold indian_red1]"
        elif "BUY" in label_upper:
            signal_str = f"[#00d700]↑ BUY[/#00d700]"
        elif "SELL" in label_upper:
            signal_str = f"[indian_red1]↓ SELL[/indian_red1]"
        else:
            signal_str = f"[#87afaf]HOLD[/#87afaf]"
        
        table.add_row(
            horizon_label,
            edge_str,
            prob_str,
            ret_str,
            ci_str,
            strength_str,
            regime_str,
            profit_str,
            signal_str,
        )

    # Create title with asset info
    panel_title = f"[bold cyan]{asset_symbol}[/bold cyan] — [white]{title}[/white]"
    panel_subtitle = f"[dim]Last: {last_close:,.2f} on {last_date}[/dim]"
    
    # Print with panel wrapper for elegance
    console.print()
    console.print(Panel(
        table,
        title=panel_title,
        subtitle=panel_subtitle,
        border_style="cyan",
        padding=(1, 2),
    ))
    
    # Add compact caption with methodology
    if show_caption:
        cdf_name = "Student-t" if used_student_t_mapping else "Normal"
        console.print(f"[dim]  Edge = risk-adjusted z-score • Prob mapped via {cdf_name} CDF • Strength = EU-based position sizing[/dim]")
        console.print(f"[dim]  Profit on 1M PLN • CI = {int(confidence_level*100)}% confidence interval • HOLD when |edge| < floor[/dim]")
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
        asset_col_width = max(28, min(40, longest_asset + 2))

    if console is None:
        console = Console()
    
    # Clean compact table with elegant styling
    table = Table(
        title=title_override,
        show_header=True,
        header_style="bold cyan",
        border_style="blue",
        padding=(0, 1),
    )
    # Asset column
    table.add_column("Asset", justify="left", style="bold", width=asset_col_width, no_wrap=False)
    
    # Compact horizon labels
    horizon_labels = {1: "1d", 3: "3d", 7: "1w", 21: "1m", 63: "3m", 126: "6m", 252: "12m"}
    for horizon in horizons:
        label = horizon_labels.get(horizon, f"{horizon}d")
        table.add_column(label, justify="center", width=19)

    for row in sorted_rows:
        asset_label = row.get("asset_label", "Unknown")
        horizon_signals = row.get("horizon_signals", {})
        
        cells = []
        for horizon in horizons:
            signal_data = horizon_signals.get(horizon) or horizon_signals.get(str(horizon)) or {}
            label = signal_data.get("label", "HOLD")
            profit_pln = signal_data.get("profit_pln", 0.0)
            cells.append(format_profit_with_signal(label, profit_pln))
        
        table.add_row(asset_label, *cells)

    console.print(table)
    console.print()


def render_sector_summary_tables(summary_rows: List[Dict], horizons: List[int]) -> None:
    """Render compact sector-grouped tables with clean visual hierarchy."""
    if not summary_rows:
        return

    buckets: Dict[str, List[Dict]] = {}
    for row in summary_rows:
        sector = row.get("sector", "") or ""
        sector = sector.strip() if sector else "Other"  # Default to "Other" for empty sectors
        buckets.setdefault(sector, []).append(row)

    import re
    def _plain_len(text: str) -> int:
        if not isinstance(text, str):
            return 0
        return len(re.sub(r"\[/?[^\]]+\]", "", text))
    longest_asset = max((_plain_len(r.get("asset_label", "")) for r in summary_rows), default=0)
    asset_col_width = max(22, min(32, longest_asset))

    console = Console(force_terminal=True)
    
    # Print header and legend with elegant styling (no background)
    console.print()
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  SIGNAL DASHBOARD[/bold cyan]                                                    [white]Returns on 1M PLN investment[/white]")
    console.print("[white]  ▲▲▼▼ Strong Signal   △▽ Notable [dim](HOLD w/ big return)[/dim]   ↑↓ Signal   [#00d700]Aqua[/#00d700]=Positive [indian_red1]Salmon[/indian_red1]=Negative[/white]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print()

    # Sort sectors alphabetically but put "Other" at the end
    def sector_sort_key(item):
        sector_name = item[0]
        if sector_name in ("Other", "Unspecified"):
            return ("~", sector_name)  # ~ sorts after letters
        return ("", sector_name)

    first_sector = True
    for sector, rows in sorted(buckets.items(), key=sector_sort_key):
        # Print sector name as a centered header with elegant orange styling
        if first_sector:
            console.print(f"\n[bold orange1]━━━━━━━━━━━━━━━━━━━━━━  {sector}  ━━━━━━━━━━━━━━━━━━━━━━[/bold orange1]", justify="center")
            first_sector = False
        else:
            console.print(f"\n[bold orange1]━━━━━━━━━━━━━━━━━━━━━━  {sector}  ━━━━━━━━━━━━━━━━━━━━━━[/bold orange1]", justify="center")
        render_multi_asset_summary_table(rows, horizons, title_override=None, asset_col_width=asset_col_width, console=console)


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
    return Console(force_terminal=True, color_system="auto")


def render_tuning_header(
    prior_mean: float,
    prior_lambda: float,
    lambda_regime: float,
    console: Console = None
) -> None:
    """Render beautiful header for tuning pipeline.
    
    Args:
        prior_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        lambda_regime: Hierarchical shrinkage parameter
        console: Rich console instance
    """
    if console is None:
        console = create_tuning_console()
    
    header_panel = Panel(
        Text.from_markup(
            f"[bold white]Kalman Drift MLE Tuning Pipeline[/bold white]\n"
            f"[dim]Hierarchical Regime-Conditional Bayesian Model Averaging[/dim]\n\n"
            f"[cyan]Prior:[/cyan] log₁₀(q) ~ N({prior_mean:.1f}, λ={prior_lambda:.1f})\n"
            f"[cyan]Hierarchical shrinkage:[/cyan] λ_regime = {lambda_regime:.3f}\n"
            f"[cyan]Model selection:[/cyan] Gaussian vs Student-t via BIC\n"
            f"[cyan]Regime-conditional:[/cyan] Fits (q, φ, ν) per market regime"
        ),
        title="[bold blue]🔧 KALMAN TUNING[/bold blue]",
        border_style="blue",
        padding=(1, 2),
        expand=False,
    )
    console.print(header_panel)


def render_tuning_progress_start(
    n_assets: int,
    n_workers: int,
    console: Console = None
) -> None:
    """Render progress start message.
    
    Args:
        n_assets: Number of assets to process
        n_workers: Number of parallel workers
        console: Rich console instance
    """
    if console is None:
        console = create_tuning_console()
    
    console.print(
        f"\n[bold cyan]🚀 Processing {n_assets} assets[/bold cyan] "
        f"[dim]({n_workers} parallel workers)[/dim]"
    )


def render_cache_status(
    cache_size: int,
    cache_path: str,
    console: Console = None
) -> None:
    """Render cache status message.
    
    Args:
        cache_size: Number of entries in cache
        cache_path: Path to cache file
        console: Rich console instance
    """
    if console is None:
        console = create_tuning_console()
    
    console.print(f"\n[dim]💾 Loaded cache with {cache_size} existing entries[/dim]")
    console.print(f"[dim]   Path: {cache_path}[/dim]")


def render_cache_update(
    cache_path: str,
    console: Console = None
) -> None:
    """Render cache update confirmation message.
    
    Args:
        cache_path: Path to cache file
        console: Rich console instance
    """
    if console is None:
        console = create_tuning_console()
    
    console.print(f"\n[#00d700]✓ Cache updated: {cache_path}[/#00d700]")


def render_asset_progress(
    asset: str,
    index: int,
    total: int,
    status: str,
    details: Optional[str] = None,
    console: Console = None
) -> None:
    """Render progress for a single asset.
    
    Args:
        asset: Asset symbol
        index: Current index (1-based)
        total: Total assets
        status: Status indicator ('success', 'cached', 'failed', 'warning')
        details: Optional detail string
        console: Rich console instance
    """
    if console is None:
        console = create_tuning_console()
    
    # Status icons and colors
    status_styles = {
        'success': ('[bold #00d700]✓[/bold #00d700]', '#00d700'),
        'cached': ('[cyan]↻[/cyan]', 'cyan'),
        'failed': ('[bold indian_red1]✗[/bold indian_red1]', 'indian_red1'),
        'warning': ('[yellow]⚠[/yellow]', 'yellow'),
        'processing': ('[blue]⟳[/blue]', 'blue'),
    }
    
    icon, color = status_styles.get(status, ('[white]•[/white]', 'white'))
    progress_pct = index / total * 100
    
    msg = f"{icon} [{color}]{asset}[/{color}]"
    if details:
        msg += f" [dim]{details}[/dim]"
    
    # Add progress indicator
    bar_width = 20
    filled = int(progress_pct / 100 * bar_width)
    bar = f"[cyan]{'█' * filled}{'░' * (bar_width - filled)}[/cyan]"
    
    console.print(f"  [{index:3d}/{total}] {bar} {msg}")


def _get_status(fit_count: int, shrunk_count: int) -> str:
    """Get plain text status for regime row."""
    if fit_count == 0:
        return "no fits"
    elif shrunk_count > 0:
        pct = shrunk_count / fit_count * 100 if fit_count > 0 else 0
        return f"{pct:.0f}% shrunk"
    else:
        return "✓ estimated"


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
    console: Console = None
) -> None:
    """Render comprehensive tuning summary with beautiful formatting.
    
    Args:
        total_assets: Total number of assets processed
        new_estimates: Number of newly estimated assets
        reused_cached: Number of assets reused from cache
        failed: Number of failed assets
        calibration_warnings: Number of calibration warnings
        gaussian_count: Number of Gaussian models
        student_t_count: Number of Student-t models
        regime_tuning_count: Number of assets with regime params
        lambda_regime: Hierarchical shrinkage parameter
        regime_fit_counts: Dict mapping regime index to fit count
        regime_shrunk_counts: Dict mapping regime index to shrunk count
        collapse_warnings: Number of collapse warnings
        cache_path: Path to cache file
        console: Rich console instance
    """
    if console is None:
        console = create_tuning_console()
    
    console.print()
    
    # Processing summary - Rich Panel with expand=False
    summary_text = (
        f"Assets processed     [bold white]{total_assets}[/bold white]\n"
        f"New estimates        [#00d700]{new_estimates}[/#00d700]\n"
        f"Reused cached        [cyan]{reused_cached}[/cyan]\n"
        f"Failed               {'[indian_red1]' + str(failed) + '[/indian_red1]' if failed > 0 else '[dim]0[/dim]'}"
    )
    if calibration_warnings > 0:
        summary_text += f"\nCalibration warnings [yellow]⚠ {calibration_warnings}[/yellow]"
    
    console.print(Panel(
        summary_text,
        title="[bold cyan]📊 Processing Summary[/bold cyan]",
        border_style="cyan",
        expand=False,
        padding=(1, 2),
    ))
    
    # Model selection summary - Rich Panel with expand=False
    total_models = gaussian_count + student_t_count
    if total_models > 0:
        gauss_pct = gaussian_count / total_models * 100
        student_pct = student_t_count / total_models * 100
        
        gauss_bar = "█" * int(gauss_pct / 5) + "░" * (20 - int(gauss_pct / 5))
        student_bar = "█" * int(student_pct / 5) + "░" * (20 - int(student_pct / 5))
        
        model_text = (
            f"Gaussian   {gaussian_count:>3}  [green]{gauss_bar}[/green] {gauss_pct:>3.0f}%\n"
            f"Student-t  {student_t_count:>3}  [magenta]{student_bar}[/magenta] {student_pct:>3.0f}%"
        )
        
        console.print(Panel(
            model_text,
            title="[bold magenta]🎯 Model Selection (BIC)[/bold magenta]",
            border_style="magenta",
            expand=False,
            padding=(1, 2),
        ))
    
    # Regime-conditional summary - using Rich Table instead of Panel
    console.print()
    console.print("[bold yellow]🌡️ Regime-Conditional Tuning[/bold yellow]")
    console.print()
    
    regime_table = Table(
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
        border_style="yellow",
        padding=(0, 1),
        expand=False,
    )
    regime_table.add_column("Regime", style="bold", width=16)
    regime_table.add_column("Fits", justify="right", width=6)
    regime_table.add_column("Shrunk", justify="right", width=6)
    regime_table.add_column("Status", width=14)
    
    for r in range(5):
        regime_name = TUNING_REGIME_LABELS.get(r, f"REGIME_{r}")
        regime_color = REGIME_COLORS.get(regime_name, "white")
        fit_count = regime_fit_counts.get(r, 0)
        shrunk_count = regime_shrunk_counts.get(r, 0)
        
        if fit_count == 0:
            status = "[dim]no fits[/dim]"
        elif shrunk_count > 0:
            pct_shrunk = shrunk_count / fit_count * 100 if fit_count > 0 else 0
            status = f"[yellow]{pct_shrunk:.0f}% shrunk[/yellow]"
        else:
            status = "[#00d700]✓ estimated[/#00d700]"
        
        shrunk_display = str(shrunk_count) if shrunk_count > 0 else "-"
        
        regime_table.add_row(
            f"[{regime_color}]{regime_name}[/{regime_color}]",
            str(fit_count),
            shrunk_display,
            status
        )
    
    console.print(regime_table)
    console.print(f"[dim]  λ_regime = {lambda_regime:.3f}  •  Regime params: {regime_tuning_count}[/dim]")
    if collapse_warnings > 0:
        console.print(f"[yellow]  ⚠ Collapse warnings: {collapse_warnings} assets[/yellow]")
    
    # Cache info
    console.print(f"\n[dim]💾 Cache: {cache_path}[/dim]")


def render_parameter_table(
    cache: Dict[str, Dict],
    console: Console = None
) -> None:
    """Render beautiful parameter table grouped by model family.
    
    Args:
        cache: Cache dictionary with asset parameters
        console: Rich console instance
    """
    if console is None:
        console = create_tuning_console()
    
    if not cache:
        console.print("[dim]No parameters to display.[/dim]")
        return
    
    def _model_label(data: dict) -> str:
        """Determine model label from cache entry."""
        if 'global' in data:
            data = data['global']
        phi_val = data.get('phi')
        noise_model = data.get('noise_model', 'gaussian')
        if noise_model in ('kalman_phi_student_t', 'phi_student_t') and phi_val is not None:
            return 'Phi-Student-t'
        if noise_model in ('kalman_phi_student_t', 'phi_student_t'):
            return 'Student-t'
        if noise_model == 'phi_gaussian' or phi_val is not None:
            return 'Phi-Gaussian'
        return 'Gaussian'
    
    def _get_q_for_sort(data):
        if 'global' in data:
            return data['global'].get('q', 0)
        return data.get('q', 0)
    
    # Sort by model family, then descending q
    sorted_assets = sorted(
        cache.items(),
        key=lambda x: (_model_label(x[1]), -_get_q_for_sort(x[1]))
    )
    
    # Group by model family
    model_groups: Dict[str, List] = {}
    for asset, raw_data in sorted_assets:
        model = _model_label(raw_data)
        if model not in model_groups:
            model_groups[model] = []
        model_groups[model].append((asset, raw_data))
    
    # Model family colors
    model_colors = {
        'Gaussian': 'green',
        'Phi-Gaussian': 'cyan',
        'Student-t': 'yellow',
        'Phi-Student-t': 'magenta',
    }
    
    console.print()
    console.print("[bold white]Best-fit Parameters by Model Family[/bold white]")
    console.print("[dim]Sorted by log₁₀(q) within each group[/dim]")
    console.print()
    
    for model_name, assets in model_groups.items():
        color = model_colors.get(model_name, 'white')
        
        table = Table(
            title=f"[bold {color}]● {model_name}[/bold {color}] [dim]({len(assets)} assets)[/dim]",
            show_header=True,
            header_style="bold white",
            box=box.ROUNDED,
            padding=(0, 1),
        )
        
        table.add_column("Asset", style="bold", width=14)
        table.add_column("log₁₀(q)", justify="right", width=9)
        table.add_column("c", justify="right", width=6)
        table.add_column("φ", justify="right", width=7)
        table.add_column("ν", justify="right", width=6)
        table.add_column("ΔLL₀", justify="right", width=7)
        table.add_column("ΔLLc", justify="right", width=7)
        table.add_column("ΔLLe", justify="right", width=7)
        table.add_column("BIC", justify="right", width=9)
        table.add_column("PIT p", justify="right", width=8)
        
        for asset, raw_data in assets:
            # Handle regime-conditional structure
            if 'global' in raw_data:
                data = raw_data['global']
            else:
                data = raw_data
            
            q_val = data.get('q', float('nan'))
            c_val = data.get('c', 1.0)
            nu_val = data.get('nu')
            phi_val = data.get('phi')
            delta_ll_zero = data.get('delta_ll_vs_zero', float('nan'))
            delta_ll_const = data.get('delta_ll_vs_const', float('nan'))
            delta_ll_ewma = data.get('delta_ll_vs_ewma', float('nan'))
            bic_val = data.get('bic', float('nan'))
            pit_p = data.get('pit_ks_pvalue', float('nan'))
            
            log10_q = np.log10(q_val) if q_val > 0 else float('nan')
            
            # Format values with color coding
            log_q_str = f"{log10_q:.2f}" if np.isfinite(log10_q) else "-"
            c_str = f"{c_val:.3f}"
            phi_str = f"{phi_val:+.3f}" if phi_val is not None else "[dim]-[/dim]"
            nu_str = f"{nu_val:.1f}" if nu_val is not None else "[dim]-[/dim]"
            
            # Color code delta LL (positive = better than baseline)
            def _format_delta_ll(val):
                if not np.isfinite(val):
                    return "[dim]-[/dim]"
                if val > 10:
                    return f"[#00d700]{val:+.0f}[/#00d700]"
                elif val > 0:
                    return f"[cyan]{val:+.0f}[/cyan]"
                elif val < -10:
                    return f"[indian_red1]{val:+.0f}[/indian_red1]"
                else:
                    return f"[dim]{val:+.0f}[/dim]"
            
            bic_str = f"{bic_val:.0f}" if np.isfinite(bic_val) else "-"
            
            # Color code PIT p-value
            if np.isfinite(pit_p):
                if pit_p < 0.01:
                    pit_str = f"[indian_red1]{pit_p:.4f}[/indian_red1]"
                elif pit_p < 0.05:
                    pit_str = f"[yellow]{pit_p:.4f}[/yellow]"
                else:
                    pit_str = f"[#00d700]{pit_p:.4f}[/#00d700]"
            else:
                pit_str = "[dim]-[/dim]"
            
            # Add warning marker if calibration warning
            if data.get('calibration_warning'):
                pit_str += " [yellow]⚠[/yellow]"
            
            table.add_row(
                asset,
                log_q_str,
                c_str,
                phi_str,
                nu_str,
                _format_delta_ll(delta_ll_zero),
                _format_delta_ll(delta_ll_const),
                _format_delta_ll(delta_ll_ewma),
                bic_str,
                pit_str,
            )
        
        console.print(table)
        console.print()
    
    # Legend - Rich Panel with expand=False
    legend_text = (
        "[white]log₁₀(q)[/white] — Process noise variance (log scale)\n"
        "[white]c[/white] — Observation noise multiplier\n"
        "[white]φ[/white] — Drift persistence (AR(1) coefficient)\n"
        "[white]ν[/white] — Student-t degrees of freedom\n"
        "[white]ΔLL₀[/white] — Improvement vs zero-drift baseline\n"
        "[white]ΔLLc[/white] — Improvement vs constant-drift baseline\n"
        "[white]ΔLLe[/white] — Improvement vs EWMA-drift baseline\n"
        "[white]BIC[/white] — Bayesian Information Criterion (lower = better)\n"
        "[white]PIT p[/white] — PIT KS p-value (≥0.05 = well-calibrated)"
    )
    console.print(Panel(
        legend_text,
        title="[dim]📖 Column Legend[/dim]",
        border_style="dim",
        expand=False,
        padding=(1, 2),
    ))


def render_failed_assets(
    failure_reasons: Dict[str, str],
    console: Console = None
) -> None:
    """Render failed assets with reasons and full tracebacks.
    
    Args:
        failure_reasons: Dict mapping asset to failure reason (may include traceback)
        console: Rich console instance
    """
    if console is None:
        console = create_tuning_console()
    
    if not failure_reasons:
        return
    
    # First show a summary table with just the error messages
    table = Table(
        title="[bold indian_red1]❌ Failed Assets[/bold indian_red1]",
        show_header=True,
        header_style="bold white",
        box=box.ROUNDED,
    )
    table.add_column("Asset", style="bold")
    table.add_column("Reason", style="indian_red1")
    
    for asset, reason in failure_reasons.items():
        # Extract just the first line (error message) for the table
        first_line = reason.split('\n')[0] if reason else "Unknown error"
        table.add_row(asset, first_line)
    
    console.print(table)
    console.print()
    
    # Then show full tracebacks for each failed asset
    has_tracebacks = any('\n' in reason for reason in failure_reasons.values())
    if has_tracebacks:
        console.print("[bold indian_red1]📋 Full Tracebacks:[/bold indian_red1]")
        console.print()
        for asset, reason in failure_reasons.items():
            if '\n' in reason:
                console.print(f"[bold]{asset}:[/bold]")
                # Print the full traceback with proper formatting
                lines = reason.split('\n')
                for line in lines[1:]:  # Skip first line (already shown in table)
                    if line.strip():
                        console.print(f"  [dim]{line}[/dim]")
                console.print()


def render_end_of_run_summary(
    processed_assets: Dict[str, Dict],
    regime_distributions: Dict[str, Dict[int, int]],
    model_comparisons: Dict[str, Dict],
    failure_reasons: Dict[str, str],
    processing_log: List[str],
    console: Console = None
) -> None:
    """Render comprehensive end-of-run summary with all collected data.
    
    This is the ultimate summary showing:
    1. Processing log (what was processed)
    2. Model comparison results per asset
    3. Regime distributions per asset
    4. Failed assets with full tracebacks
    
    Args:
        processed_assets: Full results per asset
        regime_distributions: Per-asset regime counts
        model_comparisons: Per-asset model comparison results
        failure_reasons: Failed assets with reasons/tracebacks
        processing_log: Log of what was processed
        console: Rich console instance
    """
    if console is None:
        console = create_tuning_console()
    
    console.print()
    console.print(Panel(
        "[bold white]END-OF-RUN SUMMARY[/bold white]",
        border_style="bold cyan",
        expand=False,
    ))
    
    # ==========================================================================
    # Section 1: Processing Log (what was processed)
    # ==========================================================================
    if processing_log:
        console.print()
        console.print("[bold cyan]📝 Processing Log[/bold cyan]")
        console.print("-" * 60)
        for log_entry in processing_log:
            if log_entry.startswith("✓"):
                console.print(f"  [#00d700]{log_entry}[/#00d700]")
            elif log_entry.startswith("❌"):
                console.print(f"  [indian_red1]{log_entry}[/indian_red1]")
            else:
                console.print(f"  {log_entry}")
    
    # ==========================================================================
    # Section 2: Model Comparison Results (per asset)
    # ==========================================================================
    if model_comparisons:
        console.print()
        console.print("[bold magenta]🔬 Model Comparison Results[/bold magenta]")
        console.print()
        
        for asset_name in sorted(model_comparisons.keys()):
            mc = model_comparisons[asset_name]
            model_comp = mc.get('model_comparison', {})
            selected = mc.get('selected_model', 'unknown')
            n_obs = mc.get('n_obs', 0)
            
            # Asset header
            console.print(f"  [bold white]{asset_name}[/bold white] [dim]({n_obs} obs)[/dim]")
            
            # Create a mini table for this asset's model comparison
            table = Table(
                show_header=True,
                header_style="dim",
                box=box.SIMPLE,
                padding=(0, 1),
                expand=False,
            )
            table.add_column("Model", style="dim", width=20)
            table.add_column("LL", justify="right", width=10)
            table.add_column("AIC", justify="right", width=10)
            table.add_column("BIC", justify="right", width=10)
            table.add_column("Params", justify="left", width=15)
            
            # Add each model row
            models_order = ['zero_drift', 'constant_drift', 'ewma_drift', 
                           'kalman_gaussian', 'kalman_phi_gaussian', 'kalman_phi_student_t']
            model_display_names = {
                'zero_drift': 'Zero-drift',
                'constant_drift': 'Constant-drift',
                'ewma_drift': 'EWMA-drift',
                'kalman_gaussian': 'Kalman-Gaussian',
                'kalman_phi_gaussian': 'Kalman-φ-Gaussian',
                'kalman_phi_student_t': 'Kalman-φ-Student-t',
            }
            
            for model_key in models_order:
                if model_key in model_comp:
                    m = model_comp[model_key]
                    display_name = model_display_names.get(model_key, model_key)
                    
                    # Check if this is the selected model
                    is_selected = (model_key == selected) or \
                                  (model_key == 'kalman_phi_gaussian' and selected == 'phi_gaussian') or \
                                  (model_key == 'kalman_gaussian' and selected == 'gaussian')
                    
                    style = "[bold #00d700]" if is_selected else ""
                    end_style = "[/bold #00d700]" if is_selected else ""
                    
                    # Build params string
                    params = []
                    if 'mu' in m:
                        params.append(f"μ={m['mu']:.6f}")
                    if 'phi' in m:
                        params.append(f"φ={m['phi']:+.3f}")
                    if 'nu' in m:
                        params.append(f"ν={m['nu']:.1f}")
                    params_str = ", ".join(params) if params else "-"
                    
                    ll_val = m.get('ll', float('nan'))
                    aic_val = m.get('aic', float('nan'))
                    bic_val = m.get('bic', float('nan'))
                    hyv_val = m.get('hyvarinen_score', float('nan'))
                    
                    # Format Hyvärinen score
                    hyv_str = f"{hyv_val:.1f}" if np.isfinite(hyv_val) else "-"
                    
                    if is_selected:
                        display_name = f"→ {display_name}"
                    
                    table.add_row(
                        f"{style}{display_name}{end_style}",
                        f"{style}{ll_val:.1f}{end_style}",
                        f"{style}{aic_val:.1f}{end_style}",
                        f"{style}{bic_val:.1f}{end_style}",
                        f"{style}{hyv_str}{end_style}",
                        f"{style}{params_str}{end_style}",
                    )
            
            console.print(table)
            
            # Show selected model summary with model selection method
            ll_sel = mc.get('log_likelihood', float('nan'))
            bic_sel = mc.get('bic', float('nan'))
            hyv_sel = mc.get('hyvarinen_score', float('nan'))
            model_sel_method = mc.get('model_selection_method', 'combined')
            
            # Ensure hyv_sel is numeric before checking
            try:
                hyv_sel_float = float(hyv_sel) if hyv_sel is not None else float('nan')
                hyv_summary = f", H={hyv_sel_float:.1f}" if np.isfinite(hyv_sel_float) else ""
            except (TypeError, ValueError):
                hyv_summary = ""
            method_label = {'bic': 'BIC', 'hyvarinen': 'Hyvärinen', 'combined': 'BIC+Hyvärinen'}.get(model_sel_method, model_sel_method)
            
            console.print(f"    [#00d700]Selected: {selected} (LL={ll_sel:.1f}, BIC={bic_sel:.1f}{hyv_summary}) via {method_label}[/#00d700]")
            console.print()
    
    # ==========================================================================
    # Section 3: Regime Distributions (per asset)
    # ==========================================================================
    if regime_distributions:
        console.print()
        console.print("[bold yellow]📊 Regime Distributions[/bold yellow]")
        console.print()
        
        # Create a table for regime distributions
        regime_table = Table(
            show_header=True,
            header_style="bold",
            box=box.ROUNDED,
            border_style="yellow",
            padding=(0, 1),
            expand=False,
        )
        regime_table.add_column("Asset", style="bold", width=12)
        regime_table.add_column("Total", justify="right", width=6)
        regime_table.add_column("LOW_VOL_TREND", justify="right", width=14)
        regime_table.add_column("HIGH_VOL_TREND", justify="right", width=15)
        regime_table.add_column("LOW_VOL_RANGE", justify="right", width=14)
        regime_table.add_column("HIGH_VOL_RANGE", justify="right", width=15)
        regime_table.add_column("CRISIS_JUMP", justify="right", width=12)
        
        # Aggregate counts
        aggregate = {r: 0 for r in range(5)}
        
        for asset_name in sorted(regime_distributions.keys()):
            counts = regime_distributions[asset_name]
            total = sum(counts.values())
            
            # Accumulate aggregate
            for r, c in counts.items():
                aggregate[r] += c
            
            # Format counts with percentages
            def fmt_count(r):
                c = counts.get(r, 0)
                pct = 100.0 * c / total if total > 0 else 0
                if c == 0:
                    return "[dim]-[/dim]"
                return f"{c} ({pct:.0f}%)"
            
            regime_table.add_row(
                asset_name,
                str(total),
                fmt_count(0),
                fmt_count(1),
                fmt_count(2),
                fmt_count(3),
                fmt_count(4),
            )
        
        console.print(regime_table)
        
        # Show aggregate
        total_obs = sum(aggregate.values())
        if total_obs > 0:
            console.print()
            console.print("[dim]  Aggregate across all processed assets:[/dim]")
            regime_names = ['LOW_VOL_TREND', 'HIGH_VOL_TREND', 'LOW_VOL_RANGE', 'HIGH_VOL_RANGE', 'CRISIS_JUMP']
            regime_colors = ['green', 'red', 'cyan', 'yellow', 'magenta']
            for r in range(5):
                c = aggregate[r]
                pct = 100.0 * c / total_obs
                bar_len = int(pct / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                console.print(f"    [{regime_colors[r]}]{regime_names[r]:<15}[/{regime_colors[r]}] {c:>6,} [{regime_colors[r]}]{bar}[/{regime_colors[r]}] {pct:>5.1f}%")
    
    # ==========================================================================
    # Section 4: Failed Assets with Full Tracebacks
    # ==========================================================================
    if failure_reasons:
        console.print()
        render_failed_assets(failure_reasons, console=console)
    
    # Final separator
    console.print()
    console.print("=" * 80)
    console.print()


def render_dry_run_preview(
    assets: List[str],
    max_display: int = 15,
    console: Console = None
) -> None:
    """Render dry run preview of assets to process.
    
    Args:
        assets: List of asset symbols
        max_display: Maximum assets to display
        console: Rich console instance
    """
    if console is None:
        console = create_tuning_console()
    
    console.print()
    console.print(Panel(
        Text.from_markup("[bold yellow]DRY RUN MODE[/bold yellow]\n[dim]No actual processing will occur[/dim]"),
        border_style="yellow",
    ))
    
    console.print(f"\n[bold]Would process {len(assets)} assets:[/bold]")
    
    for i, asset in enumerate(assets[:max_display], 1):
        console.print(f"  [cyan]{i:3d}.[/cyan] {asset}")
    
    if len(assets) > max_display:
        console.print(f"  [dim]... and {len(assets) - max_display} more[/dim]")
    
    console.print()


def render_cache_status(
    cache_size: int,
    cache_path: str,
    console: Console = None
) -> None:
    """Render cache status message.

    Args:
        cache_size: Number of entries in cache
        cache_path: Path to cache file
        console: Rich console instance
    """
    if console is None:
        console = create_tuning_console()

    console.print(f"[dim]💾 Loaded cache with {cache_size} existing entries[/dim]")
    console.print(f"[dim]   Path: {cache_path}[/dim]")


def render_cache_update(
    cache_path: str,
    console: Console = None
) -> None:
    """Render cache update confirmation.

    Args:
        cache_path: Path to cache file
        console: Rich console instance
    """
    if console is None:
        console = create_tuning_console()

    console.print(f"\n[#00d700]✓ Cache updated:[/#00d700] {cache_path}")


class TuningProgressTracker:
    """
    Animated progress tracker for parallel tuning using Rich Progress.

    Provides a clean, animated progress bar with:
    - Spinning indicator showing activity
    - Progress bar with completion percentage
    - Real-time elapsed time
    - Recent asset completions
    """

    def __init__(self, total_assets: int, console: Console = None):
        self.total = total_assets
        self.current = 0
        self.console = console or create_tuning_console()
        self.successes = 0
        self.cached = 0
        self.failures = 0
        self.recent_completions = []  # Last few completions to display
        self.max_recent = 5

        # Create animated progress bar
        self.progress = Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold cyan]Tuning[/bold cyan]"),
            BarColumn(bar_width=40, complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
            refresh_per_second=10,
        )
        self.task = None
        self.progress.start()
        self.task = self.progress.add_task("Processing", total=total_assets)

    def update(self, asset: str, status: str, details: Optional[str] = None):
        """Update progress for an asset with animated display."""
        self.current += 1

        if status == 'success':
            self.successes += 1
            icon = "[#00d700]✓[/#00d700]"
        elif status == 'cached':
            self.cached += 1
            icon = "[cyan]↻[/cyan]"
        elif status == 'failed':
            self.failures += 1
            icon = "[indian_red1]✗[/indian_red1]"
        else:
            icon = "[white]•[/white]"

        # Update the progress bar
        self.progress.update(self.task, advance=1)

        # Store recent completion
        completion_str = f"{icon} {asset}"
        if details:
            completion_str += f" [dim]{details}[/dim]"
        self.recent_completions.append(completion_str)
        if len(self.recent_completions) > self.max_recent:
            self.recent_completions.pop(0)

        # Print completion below progress bar
        self.console.print(f"  {completion_str}")

    def finish(self):
        """Finish the progress bar and render completion message."""
        self.progress.stop()
        self.console.print()
        self.console.print(
            f"[bold #00d700]✓ Completed:[/bold #00d700] "
            f"{self.successes} new, {self.cached} cached, {self.failures} failed"
        )


# =============================================================================
# PARALLEL PROCESSING PROGRESS DISPLAY
# =============================================================================

class ParallelTuningProgress:
    """
    Rich-based progress display for parallel asset tuning.

    Provides a clean, non-cluttered view of parallel processing with:
    - Overall progress bar
    - Live status updates for each worker
    - Summary statistics
    """

    def __init__(self, total_assets: int, n_workers: int, console: Console = None):
        self.total = total_assets
        self.n_workers = n_workers
        self.console = console or create_tuning_console()
        self.completed = 0
        self.successes = 0
        self.failures = 0
        self.current_assets = {}  # worker_id -> asset being processed

    def create_progress(self) -> Progress:
        """Create a Rich Progress instance."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}[/bold cyan]"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )

    def render_header(self):
        """Render the processing header."""
        self.console.print()
        self.console.print(f"[bold cyan]🚀 Parallel Tuning[/bold cyan]  [dim]{self.total} assets × {self.n_workers} workers[/dim]")
        self.console.print()

    def render_completion(self, asset: str, success: bool, details: str = None):
        """Render a single asset completion."""
        if success:
            self.successes += 1
            icon = "[#00d700]✓[/#00d700]"
            detail_str = f"[dim]{details}[/dim]" if details else ""
        else:
            self.failures += 1
            icon = "[indian_red1]✗[/indian_red1]"
            detail_str = f"[indian_red1]{details}[/indian_red1]" if details else ""

        self.completed += 1
        self.console.print(f"  {icon} [white]{asset:<12}[/white] {detail_str}")

    def render_summary(self):
        """Render final summary."""
        self.console.print()
        self.console.print(
            f"[bold #00d700]✓ Completed:[/bold #00d700] "
            f"[#00d700]{self.successes} succeeded[/#00d700], "
            f"[indian_red1]{self.failures} failed[/indian_red1]"
        )


@contextmanager
def parallel_tuning_progress(total_assets: int, n_workers: int, console: Console = None):
    """
    Context manager for parallel tuning with Rich progress display.

    Usage:
        with parallel_tuning_progress(100, 12) as progress:
            for result in process_assets():
                progress.update(result)
    """
    tracker = ParallelTuningProgress(total_assets, n_workers, console)
    tracker.render_header()

    with tracker.create_progress() as progress:
        task = progress.add_task("Processing", total=total_assets)

        class ProgressUpdater:
            def __init__(self, tracker, progress, task):
                self.tracker = tracker
                self.progress = progress
                self.task = task

            def update(self, asset: str, success: bool, details: str = None):
                self.tracker.render_completion(asset, success, details)
                self.progress.advance(self.task)

            def get_stats(self):
                return {
                    'completed': self.tracker.completed,
                    'successes': self.tracker.successes,
                    'failures': self.tracker.failures,
                }

        yield ProgressUpdater(tracker, progress, task)

    tracker.render_summary()


def render_parallel_start(n_assets: int, n_workers: int, console: Console = None) -> None:
    """Render the start of parallel processing."""
    if console is None:
        console = create_tuning_console()

    console.print()
    console.print(Panel(
        f"[bold white]Processing {n_assets} assets[/bold white]\n"
        f"[dim]Using {n_workers} parallel workers[/dim]",
        title="[bold cyan]🚀 Parallel Tuning[/bold cyan]",
        border_style="cyan",
        expand=False,
        padding=(1, 2),
    ))


def render_worker_status(
    asset: str,
    stage: str,
    details: str = None,
    console: Console = None
) -> None:
    """
    Render a worker status update (for verbose mode).

    Args:
        asset: Asset being processed
        stage: Current processing stage
        details: Optional details
        console: Rich console
    """
    if console is None:
        console = create_tuning_console()

    stage_icons = {
        'regime_labels': '📊',
        'global_params': '🔧',
        'gaussian': '🔧',
        'phi_gaussian': '🔧',
        'student_t': '🔧',
        'bma': '🔄',
        'complete': '✓',
        'error': '✗',
    }

    icon = stage_icons.get(stage, '•')

    if stage == 'complete':
        console.print(f"  [#00d700]{icon}[/#00d700] [bold]{asset}[/bold] [dim]{details or ''}[/dim]")
    elif stage == 'error':
        console.print(f"  [indian_red1]{icon}[/indian_red1] [bold]{asset}[/bold] [indian_red1]{details or ''}[/indian_red1]")
    else:
        console.print(f"  [dim]{icon} {asset}: {stage}[/dim]" + (f" [dim]({details})[/dim]" if details else ""))


class LiveTuningDisplay:
    """
    Live display for tuning progress with worker status.

    Shows a compact live view of:
    - Overall progress
    - Currently active workers
    - Recent completions
    """

    def __init__(self, total_assets: int, n_workers: int, console: Console = None):
        self.total = total_assets
        self.n_workers = n_workers
        self.console = console or create_tuning_console()
        self.completed = 0
        self.successes = 0
        self.failures = 0
        self.active_workers = {}  # asset -> stage
        self.recent_completions = []  # list of (asset, success, details)
        self.max_recent = 5

    def _build_display(self) -> Table:
        """Build the live display table."""
        # Main progress
        pct = self.completed / self.total * 100 if self.total > 0 else 0
        bar_filled = int(pct / 2.5)
        bar = f"[cyan]{'█' * bar_filled}{'░' * (40 - bar_filled)}[/cyan]"

        table = Table(box=None, show_header=False, padding=(0, 1), expand=False)
        table.add_column("Content")

        # Header row
        table.add_row(f"[bold cyan]🚀 Parallel Tuning[/bold cyan]  {bar}  {self.completed}/{self.total} ({pct:.0f}%)")
        table.add_row("")

        # Active workers
        if self.active_workers:
            workers_str = "  ".join([
                f"[yellow]⟳ {asset}[/yellow]"
                for asset in list(self.active_workers.keys())[:self.n_workers]
            ])
            table.add_row(f"[dim]Active:[/dim] {workers_str}")

        # Recent completions
        for asset, success, details in self.recent_completions[-self.max_recent:]:
            if success:
                table.add_row(f"  [#00d700]✓[/#00d700] {asset} [dim]{details or ''}[/dim]")
            else:
                table.add_row(f"  [indian_red1]✗[/indian_red1] {asset} [indian_red1]{details or ''}[/indian_red1]")

        return table

    def start_asset(self, asset: str, stage: str = "starting"):
        """Mark an asset as actively being processed."""
        self.active_workers[asset] = stage

    def update_asset(self, asset: str, stage: str):
        """Update the stage of an active asset."""
        if asset in self.active_workers:
            self.active_workers[asset] = stage

    def complete_asset(self, asset: str, success: bool, details: str = None):
        """Mark an asset as completed."""
        if asset in self.active_workers:
            del self.active_workers[asset]

        self.completed += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1

        self.recent_completions.append((asset, success, details))
        if len(self.recent_completions) > self.max_recent * 2:
            self.recent_completions = self.recent_completions[-self.max_recent:]

    def render(self):
        """Render the current state."""
        self.console.print(self._build_display())

    def render_final(self):
        """Render final summary."""
        self.console.print()
        self.console.print(
            f"[bold #00d700]✓ Completed:[/bold #00d700] "
            f"[#00d700]{self.successes} succeeded[/#00d700], "
            f"[indian_red1]{self.failures} failed[/indian_red1]"
        )
