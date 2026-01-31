#!/usr/bin/env python3
"""
test_ewma_covariance.py

Test script for EWMA covariance estimator.
Verifies implementation with small multi-asset portfolio.
"""

import sys
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

# Import our new portfolio utilities
from portfolio_utils import ewma_covariance, extract_pairwise_correlations

# Import data fetching from existing utilities
from ingestion.data_utils import _fetch_px_symbol


def fetch_returns_for_assets(assets: list[str], start: str = "2020-01-01", end: str = None) -> pd.DataFrame:
    """
    Fetch log returns for multiple assets and align dates.
    
    Args:
        assets: List of Yahoo Finance symbols
        start: Start date
        end: End date (None = today)
        
    Returns:
        DataFrame with assets as columns, dates as index, log returns as values
    """
    console = Console()
    
    returns_dict = {}
    
    for asset in assets:
        try:
            px = _fetch_px_symbol(asset, start, end)
            # Compute log returns
            log_px = np.log(px)
            ret = log_px.diff().dropna()
            returns_dict[asset] = ret
            console.print(f"[green]✓[/green] Fetched {asset}: {len(ret)} returns")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to fetch {asset}: {e}")
            continue
    
    if not returns_dict:
        raise RuntimeError("No assets successfully fetched")
    
    # Align all returns on common dates using concat with outer join, then drop NaNs
    df_returns = pd.concat(returns_dict.values(), axis=1, keys=returns_dict.keys(), join='outer')
    # Flatten column names if they are multi-level (from keys parameter)
    if isinstance(df_returns.columns, pd.MultiIndex):
        df_returns.columns = df_returns.columns.get_level_values(0)
    df_returns = df_returns.dropna()
    
    console.print(f"\n[cyan]Aligned returns:[/cyan] {len(df_returns)} observations across {len(df_returns.columns)} assets")
    
    return df_returns


def display_latest_covariance(ewma_result: dict) -> None:
    """Display latest covariance matrix in a Rich table."""
    console = Console()
    
    cov_matrix = ewma_result['latest_cov_matrix']
    assets = ewma_result['assets']
    annualized = ewma_result['annualized']
    
    title = "Latest EWMA Covariance Matrix"
    if annualized:
        title += " (Annualized)"
    
    table = Table(title=title)
    table.add_column("Asset", justify="left", style="cyan")
    for asset in assets:
        table.add_column(asset, justify="right")
    
    for i, asset_i in enumerate(assets):
        row = [asset_i]
        for j, asset_j in enumerate(assets):
            val = cov_matrix.iloc[i, j]
            row.append(f"{val:.6f}")
        table.add_row(*row)
    
    console.print(table)


def display_latest_correlation(ewma_result: dict) -> None:
    """Display latest correlation matrix in a Rich table."""
    console = Console()
    
    corr_matrix = ewma_result['latest_corr_matrix']
    assets = ewma_result['assets']
    
    table = Table(title="Latest EWMA Correlation Matrix")
    table.add_column("Asset", justify="left", style="cyan")
    for asset in assets:
        table.add_column(asset, justify="right")
    
    for i, asset_i in enumerate(assets):
        row = [asset_i]
        for j, asset_j in enumerate(assets):
            val = corr_matrix.iloc[i, j]
            # Color code correlations
            if i == j:
                # Diagonal (always 1.0)
                cell = "[dim]1.000[/dim]"
            elif val > 0.7:
                cell = f"[green]{val:.3f}[/green]"
            elif val > 0.3:
                cell = f"{val:.3f}"
            elif val > -0.3:
                cell = f"[yellow]{val:.3f}[/yellow]"
            else:
                cell = f"[red]{val:.3f}[/red]"
            row.append(cell)
        table.add_row(*row)
    
    console.print(table)


def display_volatilities(ewma_result: dict) -> None:
    """Display latest asset volatilities."""
    console = Console()
    
    vol_df = ewma_result['volatility']
    assets = ewma_result['assets']
    annualized = ewma_result['annualized']
    
    # Get latest volatilities
    latest_vols = vol_df.iloc[-1]
    
    title = "Latest EWMA Volatilities"
    if annualized:
        title += " (Annualized)"
    
    table = Table(title=title)
    table.add_column("Asset", justify="left", style="cyan")
    table.add_column("Volatility", justify="right")
    table.add_column("% (annualized)", justify="right")
    
    for asset in assets:
        vol = latest_vols[asset]
        vol_pct = vol * 100
        table.add_row(
            asset,
            f"{vol:.6f}",
            f"{vol_pct:.2f}%"
        )
    
    console.print(table)


def display_correlation_time_series(ewma_result: dict, asset_pairs: list[tuple[str, str]] = None) -> None:
    """Display time series of pairwise correlations."""
    console = Console()
    
    corr_ts = extract_pairwise_correlations(ewma_result, asset_pairs)
    
    if corr_ts.empty:
        console.print("[yellow]No correlation pairs to display[/yellow]")
        return
    
    # Show recent history (last 10 observations)
    recent = corr_ts.tail(10)
    
    table = Table(title="Recent Correlation Time Series (last 10 observations)")
    table.add_column("Date", justify="left", style="cyan")
    for col in recent.columns:
        table.add_column(col, justify="right")
    
    for idx, row in recent.iterrows():
        row_data = [str(idx.date())]
        for val in row:
            if np.isfinite(val):
                if val > 0.7:
                    cell = f"[green]{val:.3f}[/green]"
                elif val > 0.3:
                    cell = f"{val:.3f}"
                elif val > -0.3:
                    cell = f"[yellow]{val:.3f}[/yellow]"
                else:
                    cell = f"[red]{val:.3f}[/red]"
            else:
                cell = "N/A"
            row_data.append(cell)
        table.add_row(*row_data)
    
    console.print(table)


def main():
    console = Console()
    
    console.print("\n[bold cyan]═══════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  EWMA Covariance Estimator Test Suite  [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════[/bold cyan]\n")
    
    # Test with 4 diverse assets
    assets = [
        "PLNJPY=X",   # FX
        "GC=F",       # Gold futures
        "BTC-USD",    # Crypto
        "^GSPC",      # S&P 500 index
    ]
    
    console.print(f"[bold]Test Assets:[/bold] {', '.join(assets)}\n")
    
    # Fetch returns
    try:
        returns = fetch_returns_for_assets(assets, start="2020-01-01")
    except Exception as e:
        console.print(f"[red]Failed to fetch returns: {e}[/red]")
        sys.exit(1)
    
    console.print(f"\n[bold]Sample returns (first 5):[/bold]")
    console.print(returns.head())
    
    # Compute EWMA covariance with RiskMetrics λ=0.94
    console.print("\n[bold cyan]Computing EWMA covariance (λ=0.94, RiskMetrics standard)...[/bold cyan]\n")
    
    try:
        ewma_result = ewma_covariance(
            returns=returns,
            lambda_decay=0.94,
            min_periods=30,
            annualize=True  # Annualized for easier interpretation
        )
    except Exception as e:
        console.print(f"[red]EWMA computation failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Display results
    console.print("\n")
    display_volatilities(ewma_result)
    
    console.print("\n")
    display_latest_correlation(ewma_result)
    
    console.print("\n")
    display_latest_covariance(ewma_result)
    
    # Show correlation time series for interesting pairs
    console.print("\n")
    interesting_pairs = [
        ("PLNJPY=X", "GC=F"),      # FX vs Gold
        ("BTC-USD", "^GSPC"),      # Crypto vs Equities
        ("GC=F", "^GSPC"),         # Gold vs Equities
    ]
    display_correlation_time_series(ewma_result, interesting_pairs)
    
    # Summary statistics
    console.print("\n[bold cyan]═══════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  Summary Statistics  [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════[/bold cyan]\n")
    
    summary = Table()
    summary.add_column("Metric", justify="left", style="cyan")
    summary.add_column("Value", justify="right")
    
    summary.add_row("Lambda (decay factor)", f"{ewma_result['lambda_decay']:.2f}")
    summary.add_row("Number of assets", str(ewma_result['n_assets']))
    summary.add_row("Observations used", str(len(ewma_result['covariance'])))
    summary.add_row("Annualized", "Yes" if ewma_result['annualized'] else "No")
    
    # Average correlation (excluding diagonal)
    corr_matrix = ewma_result['latest_corr_matrix'].values
    n = corr_matrix.shape[0]
    off_diag = [corr_matrix[i, j] for i in range(n) for j in range(i+1, n)]
    avg_corr = np.mean(off_diag) if off_diag else 0.0
    summary.add_row("Average pairwise correlation", f"{avg_corr:.3f}")
    
    # Min/max correlation
    if off_diag:
        summary.add_row("Min pairwise correlation", f"{np.min(off_diag):.3f}")
        summary.add_row("Max pairwise correlation", f"{np.max(off_diag):.3f}")
    
    console.print(summary)
    
    console.print("\n[bold green]✓ EWMA covariance test completed successfully![/bold green]\n")


if __name__ == "__main__":
    main()
