"""
===============================================================================
BACKTEST TUNING — Backtest-Specific Parameter Calibration
===============================================================================

Tunes backtest-specific parameters for behavioral testing.

PURPOSE:
    This tuning stage exists ONLY to ensure fair behavioral testing —
    NOT performance optimization.

PARAMETERS TUNED:
    - Market-specific scaling factors
    - Signal normalization parameters
    - Risk-neutral execution calibration

EXPLICITLY NOT TUNED:
    - PnL-optimized parameters
    - Sharpe-maximizing parameters
    - Any parameter that would be "overfit" to backtest returns

NON-OPTIMIZATION CONSTITUTION:
    - Parameters are tuned for REPRESENTATIVENESS, not RETURNS
    - This tuning makes the backtest FAIR, not PROFITABLE

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

import json
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

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
    Sector,
    MarketCap,
)
from .backtest_data import (
    BacktestDataBundle,
    BacktestDataset,
    load_backtest_data,
)


def _load_model_for_tuning(model_name: str):
    """
    Load model class from backtest_models directory for tuning.
    
    Returns an instantiated model or None if not found.
    """
    import importlib.util
    
    models_dir = Path(__file__).parent / "backtest_models"
    model_file = models_dir / f"{model_name}.py"
    
    if not model_file.exists():
        return None
    
    try:
        module_name = f"arena.backtest_models.{model_name}"
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(module_name, model_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            else:
                return None
        
        # Find the model class
        for attr_name in dir(module):
            if attr_name.endswith("Model") or attr_name.endswith("KalmanModel"):
                cls = getattr(module, attr_name)
                if isinstance(cls, type):
                    return cls()  # Return instantiated model
        
        return None
    except Exception as e:
        print(f"Warning: Could not load model {model_name}: {e}")
        return None


# Cache for models during tuning
_TUNING_MODEL_CACHE: Dict[str, Any] = {}


# =============================================================================
# TUNED PARAMETER STRUCTURES
# =============================================================================

@dataclass
class BacktestParams:
    """
    Backtest-specific parameters for a single model-ticker combination.
    
    These parameters are tuned for BEHAVIORAL FAIRNESS, not performance.
    
    Attributes:
        model_name: Name of the model
        ticker: Ticker symbol
        
        # Scaling parameters (ensure comparable signal magnitudes)
        vol_scaling: Volatility scaling factor
        signal_normalization: Signal normalization constant
        
        # Execution parameters (ensure realistic simulation)
        slippage_estimate_bps: Estimated slippage in basis points
        fill_ratio: Estimated fill ratio (0-1)
        
        # Regime parameters (ensure regime-appropriate behavior)
        crisis_dampening: Signal dampening during crisis regimes
        low_vol_amplification: Signal amplification during low vol
        
        # Metadata
        tuned_at: Timestamp of tuning
        data_start: Start date of training data
        data_end: End date of training data
    """
    model_name: str
    ticker: str
    
    # Scaling (for fairness)
    vol_scaling: float = 1.0
    signal_normalization: float = 1.0
    
    # Execution (for realism)
    slippage_estimate_bps: float = 5.0
    fill_ratio: float = 0.95
    
    # Regime (for stability)
    crisis_dampening: float = 0.5
    low_vol_amplification: float = 1.2
    
    # Metadata
    tuned_at: str = ""
    data_start: str = ""
    data_end: str = ""
    
    # Fitted model parameters (q, c, phi, etc.)
    model_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestParams":
        if "model_params" not in data: data["model_params"] = {}
        return cls(**data)


@dataclass
class ModelTunedParams:
    """
    All tuned parameters for a single model across all tickers.
    
    Attributes:
        model_name: Model identifier
        params: Dict mapping ticker to BacktestParams
        aggregate_stats: Cross-ticker statistics
        tuned_at: Timestamp
    """
    model_name: str
    params: Dict[str, BacktestParams] = field(default_factory=dict)
    aggregate_stats: Dict[str, float] = field(default_factory=dict)
    tuned_at: str = ""
    
    def get_params(self, ticker: str) -> Optional[BacktestParams]:
        return self.params.get(ticker)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "params": {t: p.to_dict() for t, p in self.params.items()},
            "aggregate_stats": self.aggregate_stats,
            "tuned_at": self.tuned_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelTunedParams":
        params = {
            t: BacktestParams.from_dict(p) 
            for t, p in data.get("params", {}).items()
        }
        return cls(
            model_name=data["model_name"],
            params=params,
            aggregate_stats=data.get("aggregate_stats", {}),
            tuned_at=data.get("tuned_at", ""),
        )


# =============================================================================
# PARAMETER TUNING LOGIC
# =============================================================================

def _compute_vol_scaling(returns: np.ndarray, benchmark_vol: float = 0.16) -> float:
    """
    Compute volatility scaling factor.
    
    This scales signals so different-volatility assets are comparable.
    NOT an optimization — a fairness adjustment.
    
    Args:
        returns: Log returns
        benchmark_vol: Target annualized volatility (default: 16%)
        
    Returns:
        Scaling factor
    """
    ann_vol = np.std(returns) * np.sqrt(252)
    if ann_vol < 0.01:
        return 1.0
    return benchmark_vol / ann_vol


def _estimate_slippage(
    volume: np.ndarray,
    close: np.ndarray,
    market_cap: MarketCap,
) -> float:
    """
    Estimate realistic slippage based on liquidity.
    
    This makes the backtest REALISTIC, not optimistic.
    
    Args:
        volume: Trading volume
        close: Close prices
        market_cap: Market cap category
        
    Returns:
        Estimated slippage in basis points
    """
    # Dollar volume (proxy for liquidity)
    dollar_volume = np.mean(volume * close)
    
    # Base slippage by market cap
    base_slippage = {
        MarketCap.MEGA_CAP: 2.0,
        MarketCap.LARGE_CAP: 5.0,
        MarketCap.MID_CAP: 10.0,
        MarketCap.SMALL_CAP: 20.0,
    }
    
    slippage = base_slippage.get(market_cap, 10.0)
    
    # Adjust for dollar volume (more liquid = less slippage)
    if dollar_volume > 1e9:  # > $1B daily
        slippage *= 0.5
    elif dollar_volume > 1e8:  # > $100M daily
        slippage *= 0.75
    elif dollar_volume < 1e7:  # < $10M daily
        slippage *= 2.0
    
    return min(slippage, 50.0)  # Cap at 50 bps


def _estimate_fill_ratio(volume: np.ndarray, market_cap: MarketCap) -> float:
    """
    Estimate realistic fill ratio based on liquidity.
    
    Args:
        volume: Trading volume
        market_cap: Market cap category
        
    Returns:
        Estimated fill ratio (0-1)
    """
    avg_volume = np.mean(volume)
    
    # Base fill ratio by market cap
    base_fill = {
        MarketCap.MEGA_CAP: 0.99,
        MarketCap.LARGE_CAP: 0.97,
        MarketCap.MID_CAP: 0.93,
        MarketCap.SMALL_CAP: 0.85,
    }
    
    fill_ratio = base_fill.get(market_cap, 0.95)
    
    # Adjust for volume
    if avg_volume > 10e6:
        fill_ratio = min(0.99, fill_ratio + 0.02)
    elif avg_volume < 500e3:
        fill_ratio = max(0.70, fill_ratio - 0.10)
    
    return fill_ratio


def _compute_crisis_dampening(returns: np.ndarray) -> float:
    """
    Compute crisis regime signal dampening factor.
    
    During crisis, models often produce extreme signals that would
    be impossible to execute in practice. This dampening factor
    makes crisis behavior REALISTIC.
    
    Args:
        returns: Log returns
        
    Returns:
        Dampening factor (0-1, lower = more dampening)
    """
    # Identify crisis periods (extreme negative returns)
    crisis_threshold = np.percentile(returns, 1)  # Bottom 1%
    crisis_returns = returns[returns < crisis_threshold]
    
    if len(crisis_returns) < 5:
        return 0.5  # Default dampening
    
    # More extreme crisis = more dampening needed
    crisis_severity = abs(np.mean(crisis_returns)) / np.std(returns)
    
    # Map to dampening factor (more severe = lower factor)
    dampening = max(0.2, min(0.8, 1.0 - crisis_severity * 0.2))
    
    return dampening


def tune_params_for_ticker(
    model_name: str,
    dataset: BacktestDataset,
) -> BacktestParams:
    """
    Tune backtest parameters for a single model-ticker combination.
    
    This is NOT performance optimization. This is FAIRNESS calibration.
    
    Args:
        model_name: Model name
        dataset: Backtest dataset for ticker
        
    Returns:
        Tuned parameters
    """
    df = dataset.df
    returns = dataset.returns
    
    # Extract arrays
    volume = df["Volume"].values
    close = df["Close"].values
    
    # =========================================================================
    # FIT THE ACTUAL MODEL to get model-specific parameters
    # =========================================================================
    fitted_model_params = {}
    
    if model_name not in _TUNING_MODEL_CACHE:
        _TUNING_MODEL_CACHE[model_name] = _load_model_for_tuning(model_name)
    
    model = _TUNING_MODEL_CACHE.get(model_name)
    
    if model is not None and hasattr(model, 'fit'):
        try:
            vol = pd.Series(returns).rolling(20).std().fillna(0.01).values
            vol = np.maximum(vol, 0.001)
            
            fit_result = model.fit(returns, vol)
            
            fitted_model_params = {
                'q': fit_result.get('q'),
                'c': fit_result.get('c'),
                'phi': fit_result.get('phi'),
                'complex_weight': fit_result.get('complex_weight'),
                'log_likelihood': fit_result.get('log_likelihood'),
                'bic': fit_result.get('bic'),
            }
            fitted_model_params = {k: v for k, v in fitted_model_params.items() if v is not None}
        except Exception as e:
            fitted_model_params = {'fit_error': str(e)}
    
    # =========================================================================
    # Compute execution/fairness parameters
    # =========================================================================
    vol_scaling = _compute_vol_scaling(returns)
    slippage = _estimate_slippage(volume, close, dataset.market_cap)
    fill_ratio = _estimate_fill_ratio(volume, dataset.market_cap)
    crisis_dampening = _compute_crisis_dampening(returns)
    
    # Signal normalization (based on return distribution)
    signal_norm = 1.0 / (np.std(returns) * 10 + 1e-6)
    signal_norm = max(0.1, min(10.0, signal_norm))
    
    # Low vol amplification (more signal when vol is low)
    vol_percentile_20 = np.percentile(np.abs(returns), 20)
    vol_percentile_80 = np.percentile(np.abs(returns), 80)
    low_vol_amp = vol_percentile_80 / (vol_percentile_20 + 1e-6)
    low_vol_amp = max(1.0, min(2.0, low_vol_amp))
    
    return BacktestParams(
        model_name=model_name,
        ticker=dataset.ticker,
        model_params=fitted_model_params,
        vol_scaling=vol_scaling,
        signal_normalization=signal_norm,
        slippage_estimate_bps=slippage,
        fill_ratio=fill_ratio,
        crisis_dampening=crisis_dampening,
        low_vol_amplification=low_vol_amp,
        tuned_at=datetime.now().isoformat(),
        data_start=dataset.date_range[0],
        data_end=dataset.date_range[1],
    )


# =============================================================================
# STORAGE AND LOADING
# =============================================================================

def save_tuned_params(
    model_params: ModelTunedParams,
    config: BacktestConfig,
) -> str:
    """
    Save tuned parameters to immutable storage.
    
    Once saved, parameters should NOT be modified during backtest execution.
    
    Args:
        model_params: Tuned parameters for model
        config: Backtest configuration
        
    Returns:
        Path to saved file
    """
    params_dir = Path(config.params_dir)
    model_dir = params_dir / model_params.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full model params
    filepath = model_dir / "params.json"
    with open(filepath, "w") as f:
        json.dump(model_params.to_dict(), f, indent=2)
    
    # Also save per-ticker files for transparency
    for ticker, params in model_params.params.items():
        ticker_file = model_dir / f"{ticker}.json"
        with open(ticker_file, "w") as f:
            json.dump(params.to_dict(), f, indent=2)
    
    return str(filepath)


def load_tuned_params(
    model_name: str,
    config: BacktestConfig,
) -> Optional[ModelTunedParams]:
    """
    Load frozen tuned parameters.
    
    Args:
        model_name: Model name
        config: Backtest configuration
        
    Returns:
        ModelTunedParams or None if not found
    """
    params_dir = Path(config.params_dir)
    model_dir = params_dir / model_name
    filepath = model_dir / "params.json"
    
    if not filepath.exists():
        return None
    
    with open(filepath) as f:
        data = json.load(f)
    
    return ModelTunedParams.from_dict(data)


def list_tuned_models(config: BacktestConfig) -> List[str]:
    """List all models with tuned parameters."""
    params_dir = Path(config.params_dir)
    if not params_dir.exists():
        return []
    
    models = []
    for item in params_dir.iterdir():
        if item.is_dir() and (item / "params.json").exists():
            models.append(item.name)
    
    return sorted(models)


# =============================================================================
# MULTIPROCESSING HELPERS
# =============================================================================

def _serialize_datasets(data_bundle) -> Dict[str, Any]:
    """Serialize datasets for multiprocessing transfer."""
    serialized = {}
    for ticker, dataset in data_bundle.datasets.items():
        df_reset = dataset.df.reset_index()
        df_reset['Date'] = df_reset['Date'].astype(str)
        serialized[ticker] = {
            'df': df_reset.to_dict('list'),
            'sector': dataset.sector.value if dataset.sector else None,
            'market_cap': dataset.market_cap.value if dataset.market_cap else None,
            'date_range': dataset.date_range,
            'n_observations': dataset.n_observations,
            'downloaded_at': dataset.downloaded_at,
        }
    return serialized


def _tune_model_worker(args: Tuple[str, Dict[str, Any]]) -> Tuple[str, Optional[Dict], Optional[str]]:
    """Worker function for parallel model tuning. Runs in separate PROCESS."""
    import traceback
    model_name, datasets_data = args
    try:
        datasets = {}
        for ticker, data in datasets_data.items():
            df = pd.DataFrame(data['df'])
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            dataset = BacktestDataset(
                ticker=ticker,
                sector=Sector(data['sector']) if data['sector'] else Sector.TECHNOLOGY,
                market_cap=MarketCap(data['market_cap']) if data['market_cap'] else MarketCap.LARGE_CAP,
                df=df,
                n_observations=data['n_observations'],
                date_range=(data['date_range'][0], data['date_range'][1]),
                downloaded_at=data['downloaded_at'],
            )
            datasets[ticker] = dataset
        
        model_params = ModelTunedParams(
            model_name=model_name,
            tuned_at=datetime.now().isoformat(),
        )
        for ticker, dataset in datasets.items():
            params = tune_params_for_ticker(model_name, dataset)
            model_params.params[ticker] = params
        
        all_vol_scaling = [p.vol_scaling for p in model_params.params.values()]
        all_slippage = [p.slippage_estimate_bps for p in model_params.params.values()]
        model_params.aggregate_stats = {
            "mean_vol_scaling": float(np.mean(all_vol_scaling)),
            "std_vol_scaling": float(np.std(all_vol_scaling)),
            "mean_slippage_bps": float(np.mean(all_slippage)),
            "max_slippage_bps": float(np.max(all_slippage)),
            "n_tickers": len(model_params.params),
        }
        return (model_name, model_params.to_dict(), None)
    except Exception as e:
        return (model_name, None, f"{str(e)}\n{traceback.format_exc()}")


# =============================================================================
# MAIN TUNING PIPELINE
# =============================================================================

def tune_backtest_params(
    model_names: List[str],
    data_bundle: Optional[BacktestDataBundle] = None,
    config: Optional[BacktestConfig] = None,
    force: bool = False,
    n_workers: Optional[int] = None,
    parallel: bool = True,
) -> Dict[str, ModelTunedParams]:
    """
    Tune backtest parameters for all models across all tickers.
    
    This is the entry point for `make arena-backtest-tune`.
    
    USES MULTIPROCESSING for parallel tuning across models.
    
    CONSTITUTIONAL REMINDER:
        This tuning is for BEHAVIORAL FAIRNESS, not PERFORMANCE OPTIMIZATION.
        Parameters make backtests REPRESENTATIVE, not PROFITABLE.
    
    Args:
        model_names: List of models to tune
        data_bundle: Pre-loaded data bundle (optional)
        config: Backtest configuration
        force: Force re-tuning even if cached
        n_workers: Number of parallel workers (default: CPU count - 1)
        parallel: Use multiprocessing (default: True)
        
    Returns:
        Dict mapping model_name to ModelTunedParams
    """
    config = config or DEFAULT_BACKTEST_CONFIG
    
    # Load data if not provided
    if data_bundle is None:
        data_bundle = load_backtest_data(config)
    
    if not data_bundle.datasets:
        raise ValueError("No data available for tuning. Run 'make arena-backtest-data' first.")
    
    console = Console() if RICH_AVAILABLE else None
    results: Dict[str, ModelTunedParams] = {}
    
    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    # Separate cached vs to-tune models
    models_to_tune = []
    for model_name in model_names:
        if not force:
            cached = load_tuned_params(model_name, config)
            if cached is not None:
                results[model_name] = cached
                continue
        models_to_tune.append(model_name)
    
    if console:
        console.print(Panel.fit(
            "[bold cyan]Backtest Parameter Tuning[/bold cyan]\n"
            f"Models: {len(model_names)} | To tune: {len(models_to_tune)} | Cached: {len(results)}\n"
            f"Tickers: {data_bundle.n_tickers} | Workers: {n_workers if parallel else 1}\n"
            "[dim]Tuning for BEHAVIORAL FAIRNESS, not performance[/dim]",
            border_style="cyan"
        ))
        
        if results:
            console.print(f"  [dim]Loaded {len(results)} models from cache[/dim]")
    
    if not models_to_tune:
        if console:
            console.print("  [green]All models already tuned![/green]")
            _display_tuning_summary(console, results)
        return results
    
    # Serialize datasets ONCE for all workers
    serialized_data = _serialize_datasets(data_bundle)
    work_items = [(model_name, serialized_data) for model_name in models_to_tune]
    
    if parallel and len(models_to_tune) > 1:
        # =====================================================================
        # PARALLEL TUNING with ProcessPoolExecutor (multiprocessing)
        # =====================================================================
        if console:
            console.print(f"\n  [cyan]Starting PARALLEL tuning with {n_workers} processes...[/cyan]")
        
        completed = 0
        errors = []
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_tune_model_worker, item): item[0] for item in work_items}
            
            for future in as_completed(futures):
                model_name = futures[future]
                completed += 1
                
                try:
                    result_name, result_dict, error = future.result()
                    
                    if error:
                        errors.append((result_name, error))
                        if console:
                            console.print(f"  [red]✗ {result_name}: {error.split(chr(10))[0]}[/red]")
                    else:
                        model_params = ModelTunedParams.from_dict(result_dict)
                        save_tuned_params(model_params, config)
                        results[result_name] = model_params
                        
                        if console:
                            stats = model_params.aggregate_stats
                            console.print(
                                f"  [green]✓[/green] {result_name} "
                                f"[dim]({completed}/{len(models_to_tune)}) "
                                f"vol_scale={stats.get('mean_vol_scaling', 0):.2f}[/dim]"
                            )
                except Exception as e:
                    errors.append((model_name, str(e)))
                    if console:
                        console.print(f"  [red]✗ {model_name}: {e}[/red]")
        
        if errors and console:
            console.print(f"\n  [yellow]⚠ {len(errors)} models failed to tune[/yellow]")
    
    else:
        # =====================================================================
        # SEQUENTIAL TUNING (single model or parallel disabled)
        # =====================================================================
        if console:
            console.print(f"\n  [cyan]Tuning {len(models_to_tune)} models sequentially...[/cyan]")
        
        for i, model_name in enumerate(models_to_tune, 1):
            if console:
                console.print(f"  [cyan]• [{i}/{len(models_to_tune)}] Tuning {model_name}...[/cyan]")
            
            model_params = ModelTunedParams(
                model_name=model_name,
                tuned_at=datetime.now().isoformat(),
            )
            
            for ticker, dataset in data_bundle.datasets.items():
                params = tune_params_for_ticker(model_name, dataset)
                model_params.params[ticker] = params
            
            all_vol_scaling = [p.vol_scaling for p in model_params.params.values()]
            all_slippage = [p.slippage_estimate_bps for p in model_params.params.values()]
            
            model_params.aggregate_stats = {
                "mean_vol_scaling": float(np.mean(all_vol_scaling)),
                "std_vol_scaling": float(np.std(all_vol_scaling)),
                "mean_slippage_bps": float(np.mean(all_slippage)),
                "max_slippage_bps": float(np.max(all_slippage)),
                "n_tickers": len(model_params.params),
            }
            
            save_tuned_params(model_params, config)
            results[model_name] = model_params
            
            if console:
                console.print(
                    f"    └─ Tuned {len(model_params.params)} tickers, "
                    f"mean vol_scale={model_params.aggregate_stats['mean_vol_scaling']:.2f}"
                )
    
    # Summary
    if console:
        console.print()
        _display_tuning_summary(console, results)
    
    return results


def _display_tuning_summary(
    console: Console,
    results: Dict[str, ModelTunedParams],
) -> None:
    """Display tuning summary."""
    table = Table(
        title="Backtest Tuning Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Model", style="white")
    table.add_column("Tickers", justify="right")
    table.add_column("Vol Scale μ", justify="right")
    table.add_column("Slippage μ (bps)", justify="right")
    
    for model_name, params in results.items():
        stats = params.aggregate_stats
        table.add_row(
            model_name,
            str(stats.get("n_tickers", 0)),
            f"{stats.get('mean_vol_scaling', 0):.2f}",
            f"{stats.get('mean_slippage_bps', 0):.1f}",
        )
    
    console.print(table)


__all__ = [
    "BacktestParams",
    "ModelTunedParams",
    "tune_params_for_ticker",
    "tune_backtest_params",
    "save_tuned_params",
    "load_tuned_params",
    "list_tuned_models",
]
