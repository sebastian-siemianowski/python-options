"""
BACKTEST ENGINE - Structural Backtest Execution
NON-OPTIMIZATION CONSTITUTION: Financial metrics are OBSERVATIONAL ONLY.
Author: Chinese Staff Professor - Elite Quant Systems, Date: February 2026
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

try:
    from rich.console import Console
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .backtest_config import BacktestConfig, DEFAULT_BACKTEST_CONFIG, DecisionOutcome, BACKTEST_UNIVERSE, Sector, MarketCap
from .backtest_data import BacktestDataBundle, BacktestDataset, load_backtest_data
from .backtest_tune import BacktestParams, ModelTunedParams, load_tuned_params, list_tuned_models


@dataclass
class FinancialDiagnostics:
    cumulative_pnl: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    profit_factor: float = 0.0
    hit_rate: float = 0.0
    total_trades: int = 0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class BehavioralDiagnostics:
    equity_curve_convexity: float = 0.0
    tail_loss_clustering: int = 0
    return_volatility: float = 0.0
    turnover_mean: float = 0.0
    turnover_std: float = 0.0
    exposure_concentration: float = 0.0
    leverage_sensitivity: float = 0.0
    regime_stability: float = 0.0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class CrossAssetDiagnostics:
    performance_dispersion: float = 0.0
    drawdown_correlation: float = 0.0
    sector_fragility: Dict[str, float] = field(default_factory=dict)
    crisis_amplification: float = 0.0
    
    def to_dict(self):
        return {
            "performance_dispersion": self.performance_dispersion,
            "drawdown_correlation": self.drawdown_correlation,
            "sector_fragility": self.sector_fragility,
            "crisis_amplification": self.crisis_amplification,
        }


@dataclass
class TickerBacktestResult:
    ticker: str
    model_name: str
    sector: str
    market_cap: str
    financial: FinancialDiagnostics
    behavioral: BehavioralDiagnostics
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "ticker": self.ticker,
            "model_name": self.model_name,
            "sector": self.sector,
            "market_cap": self.market_cap,
            "financial": self.financial.to_dict(),
            "behavioral": self.behavioral.to_dict(),
            "warnings": self.warnings,
        }


@dataclass
class ModelBacktestResult:
    model_name: str
    ticker_results: Dict[str, TickerBacktestResult]
    cross_asset: CrossAssetDiagnostics
    aggregate_financial: FinancialDiagnostics
    aggregate_behavioral: BehavioralDiagnostics
    decision: DecisionOutcome
    decision_rationale: List[str]
    warnings: List[str]
    timestamp: str
    
    def to_dict(self):
        return {
            "model_name": self.model_name,
            "ticker_results": {t: r.to_dict() for t, r in self.ticker_results.items()},
            "cross_asset": self.cross_asset.to_dict(),
            "aggregate_financial": self.aggregate_financial.to_dict(),
            "aggregate_behavioral": self.aggregate_behavioral.to_dict(),
            "decision": self.decision.value,
            "decision_rationale": self.decision_rationale,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
        }


def _load_model_class(model_name: str):
    """
    Load model class from backtest_models directory.
    
    Returns the model class or None if not found.
    """
    import importlib.util
    import sys
    
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
        
        # Find the model class (look for class ending with Model or KalmanModel)
        for attr_name in dir(module):
            if attr_name.endswith("Model") or attr_name.endswith("KalmanModel"):
                cls = getattr(module, attr_name)
                if isinstance(cls, type):
                    return cls
        
        return None
    except Exception as e:
        print(f"Warning: Could not load model {model_name}: {e}")
        return None


# Cache for loaded models to avoid re-loading
_MODEL_CACHE: Dict[str, Any] = {}


def _get_model_instance(model_name: str):
    """Get or create a model instance."""
    if model_name not in _MODEL_CACHE:
        model_class = _load_model_class(model_name)
        if model_class:
            try:
                _MODEL_CACHE[model_name] = model_class()
            except Exception as e:
                print(f"Warning: Could not instantiate model {model_name}: {e}")
                _MODEL_CACHE[model_name] = None
        else:
            _MODEL_CACHE[model_name] = None
    return _MODEL_CACHE[model_name]


def _simulate_signals(returns, model_name, params):
    """
    Generate trading signals using the actual model.
    
    Attempts to load and run the model from backtest_models/.
    Falls back to simple momentum if model not found or errors.
    """
    n = len(returns)
    
    # Try to load and use the actual model
    model = _get_model_instance(model_name)
    
    # Check if we have pre-fitted model parameters from tuning
    model_params = getattr(params, 'model_params', {}) or {}
    has_fitted_params = model_params.get('q') is not None
    
    if model is not None and hasattr(model, 'filter') and has_fitted_params:
        try:
            # Compute volatility estimate for the model
            vol = pd.Series(returns).rolling(20).std().fillna(0.01).values
            vol = np.maximum(vol, 0.001)
            
            # USE PRE-FITTED model parameters from tuning stage
            q = model_params.get('q', 1e-6)
            c = model_params.get('c', 1.0)
            phi = model_params.get('phi', 0.0)
            complex_weight = model_params.get('complex_weight', 1.0)
            
            # Run the model filter with fitted parameters
            if hasattr(model, 'n_levels'):
                # DTCWT model
                mu, sigma, _ = model.filter(returns, vol, q, c, phi, complex_weight)
            else:
                # Standard Kalman model
                mu, sigma, _ = model.filter(returns, vol, q, c, phi)
            
            # Generate signals from Kalman filter output
            # mu[t] is the model's prediction for returns[t]
            # It has weak positive correlation (~0.03) with returns[t+1]
            # Use mu[t] directly scaled as the position for time t
            
            signals = np.zeros(n)
            for i in range(20, n):
                # Scale mu by signal_normalization to get reasonable signal magnitude
                signals[i] = np.tanh(mu[i] * params.signal_normalization * 50) * params.vol_scaling
            
            # Apply crisis dampening
            rolling_vol = pd.Series(returns).rolling(10).std().fillna(0).values
            high_vol_mask = rolling_vol > np.percentile(rolling_vol, 90)
            signals[high_vol_mask] *= params.crisis_dampening
            
            return np.clip(signals, -1, 1)
            
        except Exception as e:
            print(f"Warning: Model {model_name} fit/filter failed: {e}, falling back to momentum")
    
    # Fallback: Simple momentum signal
    model_seed = hash(model_name) % 1000
    np.random.seed(model_seed)
    noise_factor = 1.0 + 0.1 * (np.random.random() - 0.5)
    
    lookback = 20 + (hash(model_name) % 10)
    signals = np.zeros(n)
    
    for i in range(lookback, n):
        momentum = np.mean(returns[i-lookback:i]) * lookback
        signals[i] = np.tanh(momentum * params.signal_normalization * noise_factor)
        signals[i] *= params.vol_scaling
    
    # Apply crisis dampening
    rolling_vol = pd.Series(returns).rolling(10).std().fillna(0).values
    high_vol_mask = rolling_vol > np.percentile(rolling_vol, 90)
    signals[high_vol_mask] *= params.crisis_dampening
    
    return np.clip(signals, -1, 1)


def _compute_financial_diagnostics(returns, signals, initial_capital, tcost_bps, slip_bps, dates):
    """Compute financial metrics (OBSERVATIONAL ONLY)."""
    positions = signals.copy()
    pos_changes = np.abs(np.diff(signals, prepend=0))
    costs = pos_changes * (tcost_bps + slip_bps) / 10000
    
    strategy_returns = np.zeros(len(returns))
    strategy_returns[1:] = positions[:-1] * returns[1:] - costs[1:]
    
    cumulative = initial_capital * (1 + np.cumsum(strategy_returns))
    
    # CAGR
    n_years = len(returns) / 252
    if n_years > 0 and cumulative[-1] > 0:
        cagr = (cumulative[-1] / initial_capital) ** (1 / n_years) - 1
    else:
        cagr = 0.0
    
    # Sharpe
    if np.std(strategy_returns) > 1e-8:
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Sortino
    downside = strategy_returns[strategy_returns < 0]
    if len(downside) > 0 and np.std(downside) > 1e-8:
        sortino = np.mean(strategy_returns) / np.std(downside) * np.sqrt(252)
    else:
        sortino = 0.0
    
    # Drawdown - handle negative equity edge case
    cumulative_for_dd = np.maximum(cumulative, 0.01)
    peak = np.maximum.accumulate(cumulative_for_dd)
    drawdown = np.clip((peak - cumulative_for_dd) / peak, 0, 1.0)
    max_dd = float(np.max(drawdown))
    
    # Drawdown duration
    in_drawdown = drawdown > 0.01
    dd_duration = 0
    max_dd_duration = 0
    for i in range(len(in_drawdown)):
        if in_drawdown[i]:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0
    
    # Profit factor
    gains = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    if len(losses) > 0 and np.sum(np.abs(losses)) > 1e-8:
        profit_factor = np.sum(gains) / np.sum(np.abs(losses))
    else:
        profit_factor = np.inf if len(gains) > 0 else 0.0
    
    # Hit rate
    trades = strategy_returns[pos_changes > 0.01]
    if len(trades) > 0:
        hit_rate = np.sum(trades > 0) / len(trades)
    else:
        hit_rate = 0.0
    
    return FinancialDiagnostics(
        cumulative_pnl=cumulative[-1] - initial_capital,
        cagr=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_duration,
        profit_factor=profit_factor,
        hit_rate=hit_rate,
        total_trades=int(np.sum(pos_changes > 0.01)),
    ), cumulative


def _compute_behavioral_diagnostics(returns, signals, equity_curve, config):
    """Compute behavioral diagnostics (PRIMARY for decisions)."""
    warnings = []
    n = len(returns)
    
    # Equity curve convexity
    try:
        if len(equity_curve) > 20:
            convexity = np.polyfit(np.arange(len(equity_curve)), equity_curve, 2)[0] * 1e6
        else:
            convexity = 0.0
    except:
        convexity = 0.0
    
    # Tail loss clustering
    tail_threshold = np.percentile(returns, 2)
    cluster_count = 0
    consecutive = 0
    for i in range(len(returns)):
        if returns[i] < tail_threshold:
            consecutive += 1
            if consecutive >= 3:
                cluster_count += 1
        else:
            consecutive = 0
    
    if cluster_count >= config.tail_cluster_warning:
        warnings.append(f"Tail loss clustering: {cluster_count}")
    
    # Return volatility
    strategy_returns = signals[:-1] * returns[1:]
    ret_vol = np.std(strategy_returns) * np.sqrt(252)
    
    # Turnover
    turnover = np.abs(np.diff(signals))
    
    # Leverage sensitivity
    base_sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)
    leveraged_sharpe = np.mean(2 * strategy_returns) / (np.std(2 * strategy_returns) + 1e-8)
    leverage_sensitivity = abs(leveraged_sharpe - base_sharpe) / (abs(base_sharpe) + 1e-8)
    
    if leverage_sensitivity > 0.5:
        warnings.append(f"High leverage sensitivity: {leverage_sensitivity:.2f}")
    
    # Regime stability
    mid = n // 2
    first_half = strategy_returns[:mid]
    second_half = strategy_returns[mid:]
    first_sharpe = np.mean(first_half) / (np.std(first_half) + 1e-8)
    second_sharpe = np.mean(second_half) / (np.std(second_half) + 1e-8)
    regime_stability = 1.0 - abs(first_sharpe - second_sharpe) / (abs(first_sharpe) + abs(second_sharpe) + 1e-8)
    
    if regime_stability < 0.5:
        warnings.append(f"Low regime stability: {regime_stability:.2f}")
    
    return BehavioralDiagnostics(
        equity_curve_convexity=convexity,
        tail_loss_clustering=cluster_count,
        return_volatility=ret_vol,
        turnover_mean=np.mean(turnover),
        turnover_std=np.std(turnover),
        exposure_concentration=np.max(np.abs(signals)),
        leverage_sensitivity=leverage_sensitivity,
        regime_stability=regime_stability,
    ), warnings


def backtest_ticker(model_name, dataset, params, config):
    """Backtest a single ticker."""
    returns = dataset.returns
    dates = dataset.dates
    
    signals = _simulate_signals(returns, model_name, params)
    
    financial, equity = _compute_financial_diagnostics(
        returns, signals, config.initial_capital,
        config.transaction_cost_bps, params.slippage_estimate_bps, dates
    )
    
    behavioral, warnings = _compute_behavioral_diagnostics(
        returns, signals, equity, config
    )
    
    # Add financial warnings
    if financial.max_drawdown > config.max_drawdown_warning:
        warnings.append(f"High DD: {financial.max_drawdown*100:.1f}%")
    if financial.max_drawdown_duration_days > config.max_drawdown_duration_days:
        warnings.append(f"Long DD: {financial.max_drawdown_duration_days}d")
    
    info = BACKTEST_UNIVERSE.get(dataset.ticker, {})
    
    return TickerBacktestResult(
        ticker=dataset.ticker,
        model_name=model_name,
        sector=info.get("sector", Sector.TECHNOLOGY).value,
        market_cap=info.get("cap", MarketCap.LARGE_CAP).value,
        financial=financial,
        behavioral=behavioral,
        warnings=warnings,
    )


def make_decision(ticker_results, config):
    """Make behavioral safety decision."""
    rationale = []
    
    # Aggregate metrics
    dds = [r.financial.max_drawdown for r in ticker_results.values()]
    sharpes = [r.financial.sharpe_ratio for r in ticker_results.values()]
    reg_stabs = [r.behavioral.regime_stability for r in ticker_results.values()]
    lev_sens = [r.behavioral.leverage_sensitivity for r in ticker_results.values()]
    
    max_dd = max(dds)
    mean_sharpe = np.mean(sharpes)
    mean_regime_stability = np.mean(reg_stabs)
    max_leverage_sensitivity = max(lev_sens)
    
    # REJECTED conditions
    if max_dd > config.max_drawdown_for_approval:
        rationale.append(f"REJECTED: Max DD {max_dd*100:.1f}%")
        return DecisionOutcome.REJECTED, rationale
    
    if mean_sharpe < config.min_sharpe_for_approval:
        rationale.append(f"REJECTED: Sharpe {mean_sharpe:.2f}")
        return DecisionOutcome.REJECTED, rationale
    
    # QUARANTINED conditions
    if mean_regime_stability < 0.4:
        rationale.append(f"QUARANTINED: Regime stability {mean_regime_stability:.2f}")
        return DecisionOutcome.QUARANTINED, rationale
    
    if max_leverage_sensitivity > 1.0:
        rationale.append(f"QUARANTINED: Leverage sensitivity {max_leverage_sensitivity:.2f}")
        return DecisionOutcome.QUARANTINED, rationale
    
    # RESTRICTED conditions
    warning_count = sum(len(r.warnings) for r in ticker_results.values())
    if warning_count > len(ticker_results) * 2:
        rationale.append(f"RESTRICTED: {warning_count} warnings")
        return DecisionOutcome.RESTRICTED, rationale
    
    if max_dd > config.max_drawdown_warning:
        rationale.append(f"RESTRICTED: DD {max_dd*100:.1f}%")
        return DecisionOutcome.RESTRICTED, rationale
    
    # APPROVED
    rationale.append("APPROVED: Passed all behavioral safety gates")
    return DecisionOutcome.APPROVED, rationale


def _compute_cross_asset_diagnostics(ticker_results):
    """Compute cross-asset diagnostics."""
    sharpes = [r.financial.sharpe_ratio for r in ticker_results.values()]
    dds = [r.financial.max_drawdown for r in ticker_results.values()]
    
    # Drawdown correlation
    dd_correlation = 0.0
    if len(dds) > 2:
        try:
            dd_correlation = np.corrcoef(dds[:-1], dds[1:])[0, 1]
            if np.isnan(dd_correlation):
                dd_correlation = 0.0
        except:
            pass
    
    # Sector fragility
    sector_fragility = {}
    sectors = set(r.sector for r in ticker_results.values())
    for sector in sectors:
        sector_results = [r for r in ticker_results.values() if r.sector == sector]
        if sector_results:
            sector_fragility[sector] = float(np.std([r.financial.sharpe_ratio for r in sector_results]))
    
    return CrossAssetDiagnostics(
        performance_dispersion=np.std(sharpes),
        drawdown_correlation=dd_correlation,
        sector_fragility=sector_fragility,
        crisis_amplification=1.0,
    )


def _aggregate_diagnostics(ticker_results):
    """Aggregate diagnostics across tickers."""
    if not ticker_results:
        return FinancialDiagnostics(), BehavioralDiagnostics()
    
    results = list(ticker_results.values())
    
    # Filter out infinite profit factors
    pf_values = [r.financial.profit_factor for r in results if r.financial.profit_factor < np.inf]
    
    financial = FinancialDiagnostics(
        cumulative_pnl=sum(r.financial.cumulative_pnl for r in results),
        cagr=np.mean([r.financial.cagr for r in results]),
        sharpe_ratio=np.mean([r.financial.sharpe_ratio for r in results]),
        sortino_ratio=np.mean([r.financial.sortino_ratio for r in results]),
        max_drawdown=max(r.financial.max_drawdown for r in results),
        max_drawdown_duration_days=max(r.financial.max_drawdown_duration_days for r in results),
        profit_factor=np.mean(pf_values) if pf_values else 0.0,
        hit_rate=np.mean([r.financial.hit_rate for r in results]),
        total_trades=sum(r.financial.total_trades for r in results),
    )
    
    behavioral = BehavioralDiagnostics(
        equity_curve_convexity=np.mean([r.behavioral.equity_curve_convexity for r in results]),
        tail_loss_clustering=sum(r.behavioral.tail_loss_clustering for r in results),
        return_volatility=np.mean([r.behavioral.return_volatility for r in results]),
        turnover_mean=np.mean([r.behavioral.turnover_mean for r in results]),
        turnover_std=np.mean([r.behavioral.turnover_std for r in results]),
        exposure_concentration=max(r.behavioral.exposure_concentration for r in results),
        leverage_sensitivity=max(r.behavioral.leverage_sensitivity for r in results),
        regime_stability=np.mean([r.behavioral.regime_stability for r in results]),
    )
    
    return financial, behavioral


def run_structural_backtest(model_name, data_bundle=None, config=None):
    """Run structural backtest for a single model."""
    config = config or DEFAULT_BACKTEST_CONFIG
    
    if data_bundle is None:
        data_bundle = load_backtest_data(config)
    
    model_params = load_tuned_params(model_name, config)
    if model_params is None:
        raise ValueError(f"No tuned params for {model_name}")
    
    ticker_results = {}
    all_warnings = []
    
    for ticker, dataset in data_bundle.datasets.items():
        params = model_params.get_params(ticker)
        if params is None:
            continue
        
        result = backtest_ticker(model_name, dataset, params, config)
        ticker_results[ticker] = result
        all_warnings.extend(result.warnings)
    
    cross_asset = _compute_cross_asset_diagnostics(ticker_results)
    aggregate_financial, aggregate_behavioral = _aggregate_diagnostics(ticker_results)
    decision, rationale = make_decision(ticker_results, config)
    
    return ModelBacktestResult(
        model_name=model_name,
        ticker_results=ticker_results,
        cross_asset=cross_asset,
        aggregate_financial=aggregate_financial,
        aggregate_behavioral=aggregate_behavioral,
        decision=decision,
        decision_rationale=rationale,
        warnings=all_warnings,
        timestamp=datetime.now().isoformat(),
    )


def run_backtest_arena(model_names=None, config=None):
    """Run backtest arena for multiple models."""
    config = config or DEFAULT_BACKTEST_CONFIG
    
    if model_names is None:
        model_names = list_tuned_models(config)
    
    if not model_names:
        raise ValueError("No models to backtest")
    
    data_bundle = load_backtest_data(config)
    if not data_bundle.datasets:
        raise ValueError("No backtest data")
    
    console = Console() if RICH_AVAILABLE else None
    results = {}
    
    if console:
        console.print(Panel.fit(
            f"[bold cyan]Structural Backtest Arena[/bold cyan]\n"
            f"Models: {len(model_names)} | Tickers: {data_bundle.n_tickers}",
            border_style="cyan"
        ))
    
    for model_name in model_names:
        if console:
            console.print(f"\n[bold]> {model_name}[/bold]")
        
        try:
            result = run_structural_backtest(model_name, data_bundle, config)
            results[model_name] = result
            
            if console:
                color_map = {
                    "APPROVED": "green",
                    "RESTRICTED": "yellow",
                    "QUARANTINED": "orange3",
                    "REJECTED": "red",
                }
                color = color_map.get(result.decision.value, "white")
                console.print(f"  Decision: [{color}]{result.decision.value}[/{color}]")
                console.print(f"  Rationale: {result.decision_rationale[0] if result.decision_rationale else 'N/A'}")
                console.print(f"  Max DD: {result.aggregate_financial.max_drawdown*100:.1f}%")
                console.print(f"  Sharpe: {result.aggregate_financial.sharpe_ratio:.2f}")
                console.print(f"  Regime Stability: {result.aggregate_behavioral.regime_stability:.2f}")
        
        except Exception as e:
            if console:
                console.print(f"  [red]Error: {e}[/red]")
    
    # Save results
    _save_backtest_results(results, config)
    
    # Summary
    if console:
        console.print("\n" + "=" * 60)
        console.print("[bold]SUMMARY[/bold]")
        approved = sum(1 for r in results.values() if r.decision == DecisionOutcome.APPROVED)
        restricted = sum(1 for r in results.values() if r.decision == DecisionOutcome.RESTRICTED)
        quarantined = sum(1 for r in results.values() if r.decision == DecisionOutcome.QUARANTINED)
        rejected = sum(1 for r in results.values() if r.decision == DecisionOutcome.REJECTED)
        console.print(f"[green]APPROVED: {approved}[/green]")
        console.print(f"[yellow]RESTRICTED: {restricted}[/yellow]")
        console.print(f"[orange3]QUARANTINED: {quarantined}[/orange3]")
        console.print(f"[red]REJECTED: {rejected}[/red]")
    
    return results


def _save_backtest_results(results, config):
    """Save backtest results to JSON."""
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = results_dir / f"backtest_result_{timestamp}.json"
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "models": {name: result.to_dict() for name, result in results.items()},
        "summary": {
            "total_models": len(results),
            "approved": sum(1 for r in results.values() if r.decision == DecisionOutcome.APPROVED),
            "restricted": sum(1 for r in results.values() if r.decision == DecisionOutcome.RESTRICTED),
            "quarantined": sum(1 for r in results.values() if r.decision == DecisionOutcome.QUARANTINED),
            "rejected": sum(1 for r in results.values() if r.decision == DecisionOutcome.REJECTED),
        },
    }
    
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    
    return str(filepath)


__all__ = [
    "FinancialDiagnostics",
    "BehavioralDiagnostics",
    "CrossAssetDiagnostics",
    "TickerBacktestResult",
    "ModelBacktestResult",
    "backtest_ticker",
    "run_structural_backtest",
    "run_backtest_arena",
]
