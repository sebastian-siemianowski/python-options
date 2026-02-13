"""
===============================================================================
BACKTEST DATA — Multi-Sector Historical Data Pipeline
===============================================================================

Downloads and manages the canonical 50-ticker backtest universe.
Enforces strict column requirements for reproducibility.

Data Format (STRICT):
    Date,Open,High,Low,Close,Adj Close,Volume,Ticker
    
NO derived features are allowed at the data layer.

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

# Yahoo Finance for data download
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# Rich for progress display
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .backtest_config import (
    BacktestConfig,
    DEFAULT_BACKTEST_CONFIG,
    BACKTEST_UNIVERSE,
    BACKTEST_TICKERS,
    REQUIRED_COLUMNS,
    Sector,
    MarketCap,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BacktestDataset:
    """
    Container for a single ticker's backtest data.
    
    Attributes:
        ticker: Ticker symbol
        sector: Market sector
        market_cap: Market cap category
        df: DataFrame with OHLCV data (strict column format)
        n_observations: Number of data points
        date_range: (start_date, end_date) tuple
        downloaded_at: Download timestamp
    """
    ticker: str
    sector: Sector
    market_cap: MarketCap
    df: pd.DataFrame
    n_observations: int
    date_range: Tuple[str, str]
    downloaded_at: str
    
    def validate_columns(self) -> bool:
        """Validate that all required columns are present."""
        missing = set(REQUIRED_COLUMNS) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True
    
    @property
    def returns(self) -> np.ndarray:
        """Compute log returns from Adj Close prices."""
        close = self.df["Adj Close"].values
        returns = np.diff(np.log(close))
        return np.insert(returns, 0, 0.0)
    
    @property
    def dates(self) -> List[str]:
        """List of dates as strings."""
        return self.df["Date"].tolist()


@dataclass  
class BacktestDataBundle:
    """
    Complete data bundle for structural backtesting.
    
    Attributes:
        datasets: Dict mapping ticker to BacktestDataset
        coverage_report: Data coverage statistics
        download_timestamp: When bundle was created
    """
    datasets: Dict[str, BacktestDataset]
    coverage_report: Dict[str, any]
    download_timestamp: str
    
    @property
    def n_tickers(self) -> int:
        return len(self.datasets)
    
    @property
    def tickers(self) -> List[str]:
        return list(self.datasets.keys())
    
    def get_by_sector(self, sector: Sector) -> Dict[str, BacktestDataset]:
        """Get all datasets for a sector."""
        return {t: d for t, d in self.datasets.items() if d.sector == sector}
    
    def get_by_cap(self, cap: MarketCap) -> Dict[str, BacktestDataset]:
        """Get all datasets for a market cap category."""
        return {t: d for t, d in self.datasets.items() if d.market_cap == cap}


# =============================================================================
# DATA DOWNLOAD
# =============================================================================

def _download_ticker(
    ticker: str,
    lookback_years: int = 5,
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data for a single ticker.
    
    Args:
        ticker: Ticker symbol
        lookback_years: Years of historical data
        
    Returns:
        DataFrame with strict column format, or None if download fails
    """
    if not YF_AVAILABLE:
        raise ImportError("yfinance required: pip install yfinance")
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years * 365)
        
        yf_ticker = yf.Ticker(ticker)
        df = yf_ticker.history(start=start_date, end=end_date, auto_adjust=False)
        
        if df.empty or len(df) < 100:
            return None
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Standardize column names
        df = df.rename(columns={
            "Date": "Date",
            "Open": "Open", 
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Adj Close": "Adj Close",
            "Volume": "Volume",
        })
        
        # Format date as string
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        
        # Add ticker column
        df["Ticker"] = ticker
        
        # Handle missing Adj Close
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
        
        # Select only required columns in correct order
        df = df[REQUIRED_COLUMNS]
        
        return df
        
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None


def _save_ticker_data(df: pd.DataFrame, ticker: str, data_dir: Path) -> str:
    """Save ticker data to CSV file."""
    filepath = data_dir / f"{ticker}.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)


def _load_ticker_data(ticker: str, data_dir: Path) -> Optional[pd.DataFrame]:
    """Load ticker data from CSV file."""
    filepath = data_dir / f"{ticker}.csv"
    
    if not filepath.exists():
        return None
    
    try:
        df = pd.read_csv(filepath)
        
        # Validate columns
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            print(f"Warning: {ticker} missing columns {missing}")
            return None
        
        return df
    except Exception as e:
        print(f"Error loading {ticker}: {e}")
        return None


# =============================================================================
# MAIN DATA PIPELINE
# =============================================================================

def download_backtest_data(
    config: Optional[BacktestConfig] = None,
    tickers: Optional[List[str]] = None,
    force: bool = False,
) -> BacktestDataBundle:
    """
    Download historical data for the backtest universe.
    
    This is the entry point for `make arena-backtest-data`.
    
    Args:
        config: Backtest configuration
        tickers: Override tickers to download (default: use config)
        force: Force re-download even if cached
        
    Returns:
        BacktestDataBundle with all datasets
    """
    config = config or DEFAULT_BACKTEST_CONFIG
    tickers = tickers or config.tickers
    
    # Create data directory
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    datasets: Dict[str, BacktestDataset] = {}
    failed: List[str] = []
    cached: List[str] = []
    downloaded: List[str] = []
    
    console = Console() if RICH_AVAILABLE else None
    
    if console:
        console.print(Panel.fit(
            f"[bold cyan]Structural Backtest Data Pipeline[/bold cyan]\n"
            f"Tickers: {len(tickers)} | Lookback: {config.lookback_years} years\n"
            f"Directory: {config.data_dir}",
            border_style="cyan"
        ))
    
    # Download with progress
    if RICH_AVAILABLE and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing...", total=len(tickers))
            
            for ticker in tickers:
                progress.update(task, description=f"Processing {ticker}...")
                
                # Check cache first
                if not force:
                    df = _load_ticker_data(ticker, data_dir)
                    if df is not None:
                        info = BACKTEST_UNIVERSE.get(ticker, {})
                        datasets[ticker] = BacktestDataset(
                            ticker=ticker,
                            sector=info.get("sector", Sector.TECHNOLOGY),
                            market_cap=info.get("cap", MarketCap.LARGE_CAP),
                            df=df,
                            n_observations=len(df),
                            date_range=(df["Date"].iloc[0], df["Date"].iloc[-1]),
                            downloaded_at=datetime.now().isoformat(),
                        )
                        cached.append(ticker)
                        progress.advance(task)
                        continue
                
                # Download
                df = _download_ticker(ticker, config.lookback_years)
                if df is not None:
                    info = BACKTEST_UNIVERSE.get(ticker, {})
                    datasets[ticker] = BacktestDataset(
                        ticker=ticker,
                        sector=info.get("sector", Sector.TECHNOLOGY),
                        market_cap=info.get("cap", MarketCap.LARGE_CAP),
                        df=df,
                        n_observations=len(df),
                        date_range=(df["Date"].iloc[0], df["Date"].iloc[-1]),
                        downloaded_at=datetime.now().isoformat(),
                    )
                    _save_ticker_data(df, ticker, data_dir)
                    downloaded.append(ticker)
                else:
                    failed.append(ticker)
                
                progress.advance(task)
    else:
        # Fallback without Rich
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] Processing {ticker}...")
            
            if not force:
                df = _load_ticker_data(ticker, data_dir)
                if df is not None:
                    info = BACKTEST_UNIVERSE.get(ticker, {})
                    datasets[ticker] = BacktestDataset(
                        ticker=ticker,
                        sector=info.get("sector", Sector.TECHNOLOGY),
                        market_cap=info.get("cap", MarketCap.LARGE_CAP),
                        df=df,
                        n_observations=len(df),
                        date_range=(df["Date"].iloc[0], df["Date"].iloc[-1]),
                        downloaded_at=datetime.now().isoformat(),
                    )
                    cached.append(ticker)
                    continue
            
            df = _download_ticker(ticker, config.lookback_years)
            if df is not None:
                info = BACKTEST_UNIVERSE.get(ticker, {})
                datasets[ticker] = BacktestDataset(
                    ticker=ticker,
                    sector=info.get("sector", Sector.TECHNOLOGY),
                    market_cap=info.get("cap", MarketCap.LARGE_CAP),
                    df=df,
                    n_observations=len(df),
                    date_range=(df["Date"].iloc[0], df["Date"].iloc[-1]),
                    downloaded_at=datetime.now().isoformat(),
                )
                _save_ticker_data(df, ticker, data_dir)
                downloaded.append(ticker)
            else:
                failed.append(ticker)
    
    # Build coverage report
    coverage_report = {
        "total_requested": len(tickers),
        "successful": len(datasets),
        "cached": len(cached),
        "downloaded": len(downloaded),
        "failed": len(failed),
        "failed_tickers": failed,
        "sectors": {},
        "caps": {},
    }
    
    # Sector breakdown
    for sector in Sector:
        sector_tickers = [t for t, d in datasets.items() if d.sector == sector]
        coverage_report["sectors"][sector.value] = len(sector_tickers)
    
    # Market cap breakdown
    for cap in MarketCap:
        cap_tickers = [t for t, d in datasets.items() if d.market_cap == cap]
        coverage_report["caps"][cap.value] = len(cap_tickers)
    
    # Display summary
    if console:
        console.print()
        _display_coverage_summary(console, coverage_report, datasets)
    
    return BacktestDataBundle(
        datasets=datasets,
        coverage_report=coverage_report,
        download_timestamp=datetime.now().isoformat(),
    )


def _display_coverage_summary(
    console: Console,
    coverage_report: Dict,
    datasets: Dict[str, BacktestDataset],
) -> None:
    """Display coverage summary in Rich format."""
    # Main summary table
    table = Table(
        title="Data Coverage Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="white")
    table.add_column("Value", style="green", justify="right")
    
    table.add_row("Total Requested", str(coverage_report["total_requested"]))
    table.add_row("Successfully Loaded", str(coverage_report["successful"]))
    table.add_row("  └─ From Cache", str(coverage_report["cached"]))
    table.add_row("  └─ Downloaded", str(coverage_report["downloaded"]))
    if coverage_report["failed"] > 0:
        table.add_row("[red]Failed[/red]", f"[red]{coverage_report['failed']}[/red]")
    
    console.print(table)
    
    # Sector breakdown
    sector_table = Table(
        title="Sector Coverage",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold",
    )
    sector_table.add_column("Sector", style="white")
    sector_table.add_column("Count", justify="right")
    
    for sector, count in sorted(coverage_report["sectors"].items()):
        if count > 0:
            sector_table.add_row(sector.replace("_", " ").title(), str(count))
    
    console.print(sector_table)
    
    # Failed tickers warning
    if coverage_report["failed_tickers"]:
        console.print(
            f"\n[yellow]⚠ Failed tickers: {', '.join(coverage_report['failed_tickers'])}[/yellow]"
        )


def load_backtest_data(
    config: Optional[BacktestConfig] = None,
    tickers: Optional[List[str]] = None,
) -> BacktestDataBundle:
    """
    Load backtest data from cache (no download).
    
    Args:
        config: Backtest configuration
        tickers: Override tickers to load
        
    Returns:
        BacktestDataBundle with cached datasets
    """
    config = config or DEFAULT_BACKTEST_CONFIG
    tickers = tickers or config.tickers
    
    data_dir = Path(config.data_dir)
    
    datasets: Dict[str, BacktestDataset] = {}
    missing: List[str] = []
    
    for ticker in tickers:
        df = _load_ticker_data(ticker, data_dir)
        if df is not None:
            info = BACKTEST_UNIVERSE.get(ticker, {})
            datasets[ticker] = BacktestDataset(
                ticker=ticker,
                sector=info.get("sector", Sector.TECHNOLOGY),
                market_cap=info.get("cap", MarketCap.LARGE_CAP),
                df=df,
                n_observations=len(df),
                date_range=(df["Date"].iloc[0], df["Date"].iloc[-1]),
                downloaded_at="loaded_from_cache",
            )
        else:
            missing.append(ticker)
    
    coverage_report = {
        "total_requested": len(tickers),
        "successful": len(datasets),
        "cached": len(datasets),
        "downloaded": 0,
        "failed": len(missing),
        "failed_tickers": missing,
    }
    
    return BacktestDataBundle(
        datasets=datasets,
        coverage_report=coverage_report,
        download_timestamp=datetime.now().isoformat(),
    )


__all__ = [
    "BacktestDataset",
    "BacktestDataBundle",
    "download_backtest_data",
    "load_backtest_data",
]
