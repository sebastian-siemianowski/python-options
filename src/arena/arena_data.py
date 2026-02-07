"""
===============================================================================
ARENA DATA — Download and Load Benchmark Data
===============================================================================

Fetches historical price data for arena benchmark symbols and stores
in CSV format matching src/data/prices/ for consistency.

Data Format (matching src/data/prices/*.csv):
    Date,Open,High,Low,Close,Adj Close,Volume,Ticker
    2021-02-07,185.50,186.20,184.80,185.90,185.90,50000000,AAPL
    ...

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
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

from .arena_config import (
    ArenaConfig,
    ARENA_BENCHMARK_SYMBOLS,
    SYMBOL_CATEGORIES,
    CapCategory,
    DEFAULT_ARENA_CONFIG,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ArenaDataset:
    """
    Container for arena benchmark data.
    
    Attributes:
        symbol: Ticker symbol
        category: Market cap category
        df: DataFrame with OHLCV data (same format as src/data/prices/)
        statistics: Summary statistics
        downloaded_at: Download timestamp
    """
    symbol: str
    category: CapCategory
    df: pd.DataFrame
    statistics: Dict[str, float]
    downloaded_at: str
    
    @property
    def n_observations(self) -> int:
        """Number of observations."""
        return len(self.df)
    
    @property
    def returns(self) -> np.ndarray:
        """Compute log returns from Close prices."""
        close = self.df["Close"].values
        returns = np.diff(np.log(close))
        return np.insert(returns, 0, 0.0)  # First return is 0
    
    @property
    def dates(self) -> List[str]:
        """List of dates as strings."""
        return self.df["Date"].tolist()
    
    @property
    def prices(self) -> Dict[str, List[float]]:
        """Dictionary of price columns."""
        return {
            "open": self.df["Open"].tolist(),
            "high": self.df["High"].tolist(),
            "low": self.df["Low"].tolist(),
            "close": self.df["Close"].tolist(),
            "volume": self.df["Volume"].tolist(),
        }


# =============================================================================
# DATA DOWNLOAD
# =============================================================================

def _compute_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute summary statistics for price data."""
    from scipy.stats import skew, kurtosis
    
    # Compute returns
    close = df["Close"].values
    returns = np.diff(np.log(close))
    
    valid_returns = returns[~np.isnan(returns)]
    if len(valid_returns) < 10:
        return {
            "n_observations": len(df),
            "mean_return": 0.0,
            "std_return": 0.0,
            "skewness": 0.0,
            "kurtosis": 3.0,
            "min_return": 0.0,
            "max_return": 0.0,
        }
    
    return {
        "n_observations": len(df),
        "mean_return": float(np.mean(valid_returns)),
        "std_return": float(np.std(valid_returns)),
        "skewness": float(skew(valid_returns)),
        "kurtosis": float(kurtosis(valid_returns, fisher=False)),  # Excess kurtosis
        "min_return": float(np.min(valid_returns)),
        "max_return": float(np.max(valid_returns)),
    }


def _download_symbol(
    symbol: str,
    lookback_years: int = 5,
) -> Optional[ArenaDataset]:
    """
    Download data for a single symbol.
    
    Args:
        symbol: Ticker symbol
        lookback_years: Years of historical data
        
    Returns:
        ArenaDataset or None if download fails
    """
    if not YF_AVAILABLE:
        raise ImportError("yfinance required for data download: pip install yfinance")
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years * 365)
        
        # Download data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
        
        if df.empty or len(df) < 100:
            return None
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Format to match src/data/prices/ CSV format
        df = df.rename(columns={
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Adj Close": "Adj Close",
            "Volume": "Volume",
        })
        
        # Ensure Date is string format YYYY-MM-DD
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        
        # Add Ticker column
        df["Ticker"] = symbol
        
        # Select and order columns to match src/data/prices/ format
        columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
        
        # Handle missing Adj Close (use Close if not present)
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
        
        df = df[columns]
        
        # Compute statistics
        statistics = _compute_statistics(df)
        
        # Get category
        category = SYMBOL_CATEGORIES.get(symbol, CapCategory.MID_CAP)
        
        return ArenaDataset(
            symbol=symbol,
            category=category,
            df=df,
            statistics=statistics,
            downloaded_at=datetime.now().isoformat(),
        )
        
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return None


def _save_to_csv(dataset: ArenaDataset, data_dir: Path) -> str:
    """
    Save dataset to CSV file matching src/data/prices/ format.
    
    Args:
        dataset: ArenaDataset to save
        data_dir: Directory to save to
        
    Returns:
        Path to saved file
    """
    # Normalize symbol for filename (same as src/data/prices/)
    filename = f"{dataset.symbol}_1d.csv"
    filepath = data_dir / filename
    
    # Save to CSV
    dataset.df.to_csv(filepath, index=False)
    
    return str(filepath)


def _load_from_csv(symbol: str, data_dir: Path) -> Optional[ArenaDataset]:
    """
    Load dataset from CSV file.
    
    Args:
        symbol: Ticker symbol
        data_dir: Directory to load from
        
    Returns:
        ArenaDataset or None if file doesn't exist
    """
    filename = f"{symbol}_1d.csv"
    filepath = data_dir / filename
    
    if not filepath.exists():
        return None
    
    try:
        df = pd.read_csv(filepath)
        
        # Compute statistics
        statistics = _compute_statistics(df)
        
        # Get category
        category = SYMBOL_CATEGORIES.get(symbol, CapCategory.MID_CAP)
        
        return ArenaDataset(
            symbol=symbol,
            category=category,
            df=df,
            statistics=statistics,
            downloaded_at=datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
        )
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
        return None


def download_arena_data(
    config: Optional[ArenaConfig] = None,
    symbols: Optional[List[str]] = None,
    force: bool = False,
) -> Dict[str, ArenaDataset]:
    """
    Download data for all arena benchmark symbols.
    
    Args:
        config: Arena configuration (uses default if None)
        symbols: Specific symbols to download (uses config.symbols if None)
        force: Force re-download even if cached
        
    Returns:
        Dictionary mapping symbol to ArenaDataset
    """
    config = config or DEFAULT_ARENA_CONFIG
    symbols = symbols or config.symbols
    
    # Create data directory
    data_dir = Path(config.arena_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    datasets: Dict[str, ArenaDataset] = {}
    
    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel.fit(
            f"[bold cyan]Arena Data Download[/bold cyan]\n"
            f"Symbols: {len(symbols)} | Lookback: {config.lookback_years} years",
            border_style="cyan"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=len(symbols))
            
            for symbol in symbols:
                progress.update(task, description=f"Downloading {symbol}...")
                
                # Check cache (CSV file)
                if not force:
                    dataset = _load_from_csv(symbol, data_dir)
                    if dataset:
                        datasets[symbol] = dataset
                        progress.advance(task)
                        continue
                
                # Download
                dataset = _download_symbol(symbol, config.lookback_years)
                if dataset:
                    datasets[symbol] = dataset
                    # Save to CSV
                    _save_to_csv(dataset, data_dir)
                
                progress.advance(task)
    else:
        # Fallback without Rich
        for i, symbol in enumerate(symbols):
            print(f"[{i+1}/{len(symbols)}] Downloading {symbol}...")
            
            if not force:
                dataset = _load_from_csv(symbol, data_dir)
                if dataset:
                    datasets[symbol] = dataset
                    continue
            
            dataset = _download_symbol(symbol, config.lookback_years)
            if dataset:
                datasets[symbol] = dataset
                _save_to_csv(dataset, data_dir)
    
    # Print summary
    if RICH_AVAILABLE:
        table = Table(title="Arena Data Summary", box=box.ROUNDED)
        table.add_column("Category", style="cyan")
        table.add_column("Symbols", justify="right")
        table.add_column("Avg Obs", justify="right")
        table.add_column("Avg Vol", justify="right")
        
        for category in CapCategory:
            cat_datasets = [d for d in datasets.values() if d.category == category]
            if cat_datasets:
                avg_obs = np.mean([d.n_observations for d in cat_datasets])
                avg_vol = np.mean([d.statistics["std_return"] for d in cat_datasets]) * np.sqrt(252) * 100
                table.add_row(
                    category.value,
                    str(len(cat_datasets)),
                    f"{avg_obs:.0f}",
                    f"{avg_vol:.1f}%"
                )
        
        console.print(table)
    
    return datasets


def load_arena_data(
    config: Optional[ArenaConfig] = None,
    symbols: Optional[List[str]] = None,
) -> Dict[str, ArenaDataset]:
    """
    Load cached arena data (download if not cached).
    
    Args:
        config: Arena configuration
        symbols: Specific symbols to load
        
    Returns:
        Dictionary mapping symbol to ArenaDataset
    """
    config = config or DEFAULT_ARENA_CONFIG
    symbols = symbols or config.symbols
    
    data_dir = Path(config.arena_data_dir)
    datasets: Dict[str, ArenaDataset] = {}
    missing: List[str] = []
    
    for symbol in symbols:
        dataset = _load_from_csv(symbol, data_dir)
        if dataset:
            datasets[symbol] = dataset
        else:
            missing.append(symbol)
    
    # Download missing data
    if missing:
        print(f"Downloading {len(missing)} missing symbols...")
        new_datasets = download_arena_data(config, symbols=missing)
        datasets.update(new_datasets)
    
    return datasets
