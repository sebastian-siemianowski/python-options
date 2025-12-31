#!/usr/bin/env python3
"""
fx_signals_presentation.py

Presentation layer for FX signals - handles all output formatting and display logic.
Separates presentation concerns from core signal computation logic for better modularity.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table


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


def format_profit_with_signal(signal_label: str, profit_pln: float) -> str:
    """Format signal label with profit in PLN, applying color based on profit sign.
    
    Args:
        signal_label: BUY/HOLD/SELL or STRONG BUY/SELL
        profit_pln: Expected profit in PLN
        
    Returns:
        Rich-formatted string like "BUY (+12,345)" with appropriate colors
    """
    profit_txt = f"{profit_pln:+,.0f}"
    
    # Color by profit sign
    if np.isfinite(profit_pln) and profit_pln > 0:
        profit_txt = f"[green]{profit_txt}[/green]"
    elif np.isfinite(profit_pln) and profit_pln < 0:
        profit_txt = f"[red]{profit_txt}[/red]"
    
    # Emphasize STRONG labels
    label = signal_label
    if isinstance(signal_label, str) and signal_label.upper().startswith("STRONG "):
        label = f"[bold]{signal_label}[/bold]"
    
    return f"{label} ({profit_txt})"


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
    """Render detailed signal analysis table with risk metrics.
    
    Args:
        asset_symbol: Trading symbol
        title: Full descriptive title
        signals: List of Signal dataclass instances
        price_series: Historical price series
        confidence_level: Two-sided confidence level (e.g., 0.68)
        used_student_t_mapping: Whether Student-t CDF was used for probabilities
        show_caption: Whether to show detailed column explanations
    """
    console = Console()
    last_close = convert_to_float(price_series.iloc[-1])

    table = Table(title=f"{asset_symbol} — {title} — last close {last_close:.4f} on {price_series.index[-1].date()}")
    
    # Column headers
    table.add_column("Horizon (trading days)", justify="right")
    table.add_column("Edge z (risk-adjusted)", justify="right")
    table.add_column("Pr[return>0]", justify="right")
    table.add_column("E[log return]", justify="right")
    table.add_column(f"CI ±{int(confidence_level*100)}% (log)", justify="right")
    table.add_column("Position strength (0–1)", justify="right")
    table.add_column("Regime", justify="left")
    table.add_column("Profit on 1,000,000 PLN (PLN)", justify="right")
    table.add_column("Signal", justify="center")
    
    # Add caption with methodology explanation
    if show_caption:
        cdf_name = "Student-t" if used_student_t_mapping else "Normal"
        table.caption = (
            "Edge z = (expected log return / realized vol) scaled to horizon; "
            f"Pr[return>0] mapped from Edge z via {cdf_name} CDF using a single globally fitted Student‑t tail (ν) per asset; "
            f"E[log return] sums daily drift; CI is two-sided {int(confidence_level*100)}% band for log return (log domain). "
            "Volatility is modeled via GARCH(1,1) (MLE) with EWMA fallback if unavailable. "
            "Position strength uses a half‑Kelly sizing heuristic (0–1) for real‑world robustness. "
            "Profit assumes investing 1,000,000 PLN; profit CI is exp-mapped from the log-return CI into PLN. "
            "A minimum edge floor is enforced to reduce churn: if |edge| < EDGE_FLOOR, action is HOLD. BUY = long PLN vs JPY."
        )

    for signal in signals:
        table.add_row(
            str(signal.horizon_days),
            f"{signal.score:+.2f}",
            f"{100*signal.p_up:5.1f}%",
            f"{signal.exp_ret:+.4f}",
            f"[{signal.ci_low:+.4f}, {signal.ci_high:+.4f}]",
            f"{signal.position_strength:.2f}",
            signal.regime,
            f"{signal.profit_pln:,.0f} [ {signal.profit_ci_low_pln:,.0f} .. {signal.profit_ci_high_pln:,.0f} ]",
            signal.label,
        )

    console.print(table)


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

    table = Table(title=f"{asset_symbol} — {title} — Last price {last_close:.4f} on {price_series.index[-1].date()}")
    table.add_column("Timeframe", justify="left")
    table.add_column("Chance it goes up", justify="right")
    table.add_column("Recommendation", justify="center")
    table.add_column("Why (plain English)", justify="left")
    table.caption = "Simple view: chance is based on our model today. BUY means we expect the price to rise over the timeframe."

    explanations: List[str] = []
    for signal in signals:
        explanation = generate_signal_explanation(
            signal.score, signal.p_up, trend_desc, momentum_desc, volatility_desc
        )
        explanations.append(explanation)
        table.add_row(
            format_horizon_label(signal.horizon_days),
            f"{signal.p_up*100:5.1f}%",
            signal.label,
            explanation,
        )

    console.print(table)
    return explanations


def render_multi_asset_summary_table(summary_rows: List[Dict], horizons: List[int]) -> None:
    """Render compact summary table comparing signals across multiple assets.
    
    Columns: "Ticker (name)", then one column per trading-day horizon.
    Each cell shows "Signal (±Profit)" for a 1,000,000 PLN notional.
    Sorted so SELL assets come first, then HOLD, then BUY, then STRONG BUY.
    
    Args:
        summary_rows: List of dictionaries containing asset summaries
        horizons: List of horizon days to display as columns
    """
    if not summary_rows:
        return

    # Define sort priority: SELL first, then HOLD, then BUY, then STRONG BUY
    def signal_sort_key(row: Dict) -> int:
        """Lower number = appears first in table."""
        nearest_label = row.get("nearest_label", "HOLD")
        label_upper = str(nearest_label).upper()
        if "SELL" in label_upper:
            return 0  # SELL comes first
        elif "HOLD" in label_upper:
            return 1
        elif "STRONG" in label_upper and "BUY" in label_upper:
            return 3  # STRONG BUY comes last
        elif "BUY" in label_upper:
            return 2
        else:
            return 1  # default to HOLD priority

    sorted_rows = sorted(summary_rows, key=signal_sort_key)

    console = Console()
    table = Table(title="Multi-Asset Signal Summary (1,000,000 PLN notional)")
    table.add_column("Asset", justify="left", style="bold")
    
    for horizon in horizons:
        table.add_column(format_horizon_label(horizon), justify="center")

    for row in sorted_rows:
        asset_label = row.get("asset_label", "Unknown")
        horizon_signals = row.get("horizon_signals", {})
        
        cells = []
        for horizon in horizons:
            signal_data = horizon_signals.get(horizon, {})
            label = signal_data.get("label", "HOLD")
            profit_pln = signal_data.get("profit_pln", 0.0)
            cells.append(format_profit_with_signal(label, profit_pln))
        
        table.add_row(asset_label, *cells)

    console.print(table)


def render_portfolio_allocation_table(
    portfolio_result: Dict,
    horizon_days: int,
    notional_pln: float = 1_000_000.0
) -> None:
    """
    Render Kelly portfolio allocation table showing optimal weights.
    
    Displays:
    - Asset allocations (Kelly weights)
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
        title=f"📊 Kelly Portfolio Allocation ({format_horizon_label(horizon_days)} horizon)",
        show_header=True,
        header_style="bold cyan"
    )
    alloc_table.add_column("Asset", justify="left", style="bold")
    alloc_table.add_column("Kelly Weight", justify="right")
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
        
        # Color code weights
        if weight > 0.15:
            weight_str = f"[bold green]{weight:+7.2%}[/bold green]"
        elif weight > 0.05:
            weight_str = f"[green]{weight:+7.2%}[/green]"
        elif weight < -0.05:
            weight_str = f"[red]{weight:+7.2%}[/red]"
        else:
            weight_str = f"{weight:+7.2%}"
        
        # Position size formatting
        if abs(position_pln) >= 1_000_000:
            pos_str = f"{position_pln:+,.0f}"
        else:
            pos_str = f"{position_pln:+,.0f}"
        
        # Expected return formatting
        ret_str = f"{exp_ret:+.4f}"
        if exp_ret > 0.01:
            ret_str = f"[green]{ret_str}[/green]"
        elif exp_ret < -0.01:
            ret_str = f"[red]{ret_str}[/red]"
        
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
                # Color code correlations
                if abs(corr) > 0.7:
                    corr_str = f"[red]{corr:+.2f}[/red]"
                elif abs(corr) > 0.4:
                    corr_str = f"[yellow]{corr:+.2f}[/yellow]"
                else:
                    corr_str = f"[green]{corr:+.2f}[/green]"
            row_values.append(corr_str)
        corr_table.add_row(*row_values)
    
    console.print(corr_table)
    
    # Add explanatory caption
    console.print(
        "\n[dim]💡 Kelly Criterion: Maximizes log-wealth growth while managing risk via covariance matrix.[/dim]"
    )
    console.print(
        "[dim]   Weights computed as w = (1/2) × Σ⁻¹ × μ where Σ is EWMA covariance (λ=0.94).[/dim]"
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
