#!/usr/bin/env python3
"""
===============================================================================
BACKTEST CLI — Command-Line Interface for Structural Backtest Arena
===============================================================================

Usage:
    python src/arena/backtest_cli.py data [--force]
    python src/arena/backtest_cli.py tune [--models MODEL1,MODEL2]
    python src/arena/backtest_cli.py run [--models MODEL1,MODEL2]
    python src/arena/backtest_cli.py results

NON-OPTIMIZATION CONSTITUTION:
    This CLI provides access to behavioral validation infrastructure.
    Financial metrics are OBSERVATIONAL ONLY — not for optimization.

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


def discover_backtest_models():
    """Discover models in backtest_models directory."""
    models_dir = Path(__file__).parent / "backtest_models"
    if not models_dir.exists():
        return []
    
    models = []
    for f in models_dir.glob("*.py"):
        if f.name.startswith("_"):
            continue
        models.append(f.stem)
    return sorted(models)


def cmd_data(args):
    """Download backtest data for the canonical 50-ticker universe."""
    from arena.backtest_data import download_backtest_data
    from arena.backtest_config import BacktestConfig, BACKTEST_TICKERS
    
    config = BacktestConfig()
    
    if args.tickers:
        config.tickers = [t.strip().upper() for t in args.tickers.split(",")]
    
    if args.lookback:
        config.lookback_years = args.lookback
    
    print(f"Downloading backtest data for {len(config.tickers)} tickers...")
    print(f"Lookback: {config.lookback_years} years")
    print(f"Directory: {config.data_dir}")
    print()
    
    bundle = download_backtest_data(config, force=args.force)
    
    print()
    print(f"Successfully loaded: {bundle.n_tickers} tickers")
    
    if bundle.coverage_report.get("failed_tickers"):
        print(f"Failed: {', '.join(bundle.coverage_report['failed_tickers'])}")


def cmd_tune(args):
    """Tune backtest-specific parameters."""
    from arena.backtest_tune import tune_backtest_params, list_tuned_models
    from arena.backtest_data import load_backtest_data
    from arena.backtest_config import BacktestConfig
    
    config = BacktestConfig()
    
    # Determine models to tune
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        # Discover models from backtest_models directory
        model_names = discover_backtest_models()
        if not model_names:
            print("No models found in src/arena/backtest_models/")
            print("Copy a model file there first, e.g.:")
            print("  cp src/arena/safe_storage/dualtree_complex_wavelet.py src/arena/backtest_models/")
            return
    
    print(f"Tuning backtest parameters for {len(model_names)} models...")
    print(f"Models: {', '.join(model_names)}")
    print("Note: This tunes for BEHAVIORAL FAIRNESS, not performance optimization.")
    print()
    
    # Load data
    data_bundle = load_backtest_data(config)
    
    if not data_bundle.datasets:
        print("Error: No backtest data found. Run 'make arena-backtest-data' first.")
        return
    
    # Determine parallelization
    parallel = not getattr(args, 'no_parallel', False)
    n_workers = getattr(args, 'workers', None)
    
    # Tune
    results = tune_backtest_params(
        model_names=model_names,
        data_bundle=data_bundle,
        config=config,
        force=args.force,
        parallel=parallel,
        n_workers=n_workers,
    )
    
    print()
    print(f"Tuned {len(results)} models")


def cmd_run(args):
    """Run structural backtest arena."""
    from arena.backtest_engine import run_backtest_arena
    from arena.backtest_tune import list_tuned_models
    from arena.backtest_config import BacktestConfig
    
    config = BacktestConfig()
    
    # Determine models to backtest
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        # Use tuned models (must have been tuned first)
        model_names = list_tuned_models(config)
        if not model_names:
            print("No tuned models found. Run 'make arena-backtest-tune' first.")
            return
    
    try:
        results = run_backtest_arena(
            model_names=model_names,
            config=config,
        )
        
    except ValueError as e:
        print(f"Error: {e}")
        return


def cmd_results(args):
    """Show latest backtest results with comprehensive diagnostics."""
    from pathlib import Path
    import json
    
    try:
        from rich.console import Console
        RICH_AVAILABLE = True
    except ImportError:
        RICH_AVAILABLE = False
    
    results_dir = Path("src/arena/data/backtest_results")
    if not results_dir.exists():
        print("No backtest results found. Run 'make arena-backtest' first.")
        return
    
    # Find latest result file
    result_files = sorted(results_dir.glob("backtest_result_*.json"), reverse=True)
    if not result_files:
        print("No backtest results found. Run 'make arena-backtest' first.")
        return
    
    latest = result_files[0]
    with open(latest) as f:
        data = json.load(f)
    
    if RICH_AVAILABLE:
        console = Console()
        
        # Clean header
        console.print()
        console.print("[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")
        console.print()
        console.print("  [bold white]STRUCTURAL BACKTEST RESULTS[/bold white]")
        console.print(f"  [dim]{latest.name}[/dim]")
        console.print(f"  [dim]{data['timestamp'][:19]}[/dim]")
        console.print()
        console.print("[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")
        
        decision_styles = {
            "APPROVED": ("green", "✓"),
            "RESTRICTED": ("yellow", "◐"),
            "QUARANTINED": ("orange3", "◑"),
            "REJECTED": ("red", "✗"),
        }
        
        # Display each model
        for model_name, model_data in data.get("models", {}).items():
            decision = model_data.get("decision", "UNKNOWN")
            color, icon = decision_styles.get(decision, ("white", "?"))
            
            fin = model_data.get("aggregate_financial", {})
            beh = model_data.get("aggregate_behavioral", {})
            cross = model_data.get("cross_asset", {})
            rationale = model_data.get("decision_rationale", [""])[0] if model_data.get("decision_rationale") else ""
            
            console.print()
            console.print(f"  [bold white]{model_name}[/bold white]")
            console.print(f"  [{color}]{icon} {decision}[/{color}]  [dim]{rationale}[/dim]")
            console.print()
            
            # Financial
            console.print("  [bold dim]FINANCIAL[/bold dim]")
            console.print()
            
            pnl = fin.get('cumulative_pnl', 0)
            cagr = fin.get('cagr', 0)
            sharpe = fin.get('sharpe_ratio', 0)
            sortino = fin.get('sortino_ratio', 0)
            max_dd = fin.get('max_drawdown', 0)
            dd_days = fin.get('max_drawdown_duration_days', 0)
            pf = fin.get('profit_factor', 0)
            hit_rate = fin.get('hit_rate', 0)
            trades = fin.get('total_trades', 0)
            
            pnl_color = "green" if pnl >= 0 else "red"
            cagr_color = "green" if cagr >= 0 else "red"
            sharpe_color = "green" if sharpe >= 0 else "red"
            
            console.print(f"  [dim]PnL[/dim]              [{pnl_color}]${pnl:>11,.0f}[/{pnl_color}]    [dim]CAGR[/dim]           [{cagr_color}]{cagr*100:>8.1f}%[/{cagr_color}]")
            console.print(f"  [dim]Sharpe[/dim]           [{sharpe_color}]{sharpe:>12.2f}[/{sharpe_color}]    [dim]Sortino[/dim]        [{sharpe_color}]{sortino:>8.2f}[/{sharpe_color}]")
            console.print(f"  [dim]Max Drawdown[/dim]     [red]{max_dd*100:>11.1f}%[/red]    [dim]DD Duration[/dim]    {dd_days:>7}d")
            console.print(f"  [dim]Profit Factor[/dim]    {pf:>12.2f}    [dim]Hit Rate[/dim]       {hit_rate*100:>7.1f}%")
            console.print(f"  [dim]Total Trades[/dim]     {trades:>12,}")
            console.print()
            
            # Behavioral
            console.print("  [bold dim]BEHAVIORAL[/bold dim]")
            console.print()
            
            stability = beh.get('regime_stability', 0)
            vol = beh.get('return_volatility', 0)
            turnover = beh.get('turnover_mean', 0)
            tail = beh.get('tail_loss_clustering', 0)
            conc = beh.get('exposure_concentration', 0)
            lev = beh.get('leverage_sensitivity', 0)
            
            stability_color = "green" if stability >= 0.5 else "yellow" if stability >= 0.3 else "red"
            
            console.print(f"  [dim]Regime Stability[/dim] [{stability_color}]{stability:>12.2f}[/{stability_color}]    [dim]Volatility[/dim]     {vol*100:>7.1f}%")
            console.print(f"  [dim]Turnover[/dim]         {turnover:>12.2f}    [dim]Tail Clusters[/dim]  {tail:>8}")
            console.print(f"  [dim]Concentration[/dim]    {conc:>12.2f}    [dim]Leverage Sens[/dim]  {lev:>8.2f}")
            console.print()
            
            # Cross-Asset
            console.print("  [bold dim]CROSS-ASSET[/bold dim]")
            console.print()
            
            disp = cross.get('performance_dispersion', 0)
            dd_corr = cross.get('drawdown_correlation', 0)
            crisis = cross.get('crisis_amplification', 0)
            
            console.print(f"  [dim]Dispersion[/dim]       {disp:>12.2f}    [dim]DD Correlation[/dim] {dd_corr:>8.2f}")
            console.print(f"  [dim]Crisis Amp[/dim]       {crisis:>12.2f}")
            console.print()
            
            # Sector Risk - compact
            sector_frag = cross.get('sector_fragility', {})
            if sector_frag:
                console.print("  [bold dim]SECTOR RISK[/bold dim]")
                console.print()
                
                high_risk = [(s, f) for s, f in sector_frag.items() if f > 0.7]
                moderate = [(s, f) for s, f in sector_frag.items() if 0.4 < f <= 0.7]
                low_risk = [(s, f) for s, f in sector_frag.items() if f <= 0.4]
                
                if high_risk:
                    sectors = ", ".join([s for s, _ in sorted(high_risk, key=lambda x: -x[1])])
                    console.print(f"  [red]● High[/red]       {sectors}")
                if moderate:
                    sectors = ", ".join([s for s, _ in sorted(moderate, key=lambda x: -x[1])])
                    console.print(f"  [yellow]● Moderate[/yellow]   {sectors}")
                if low_risk:
                    sectors = ", ".join([s for s, _ in sorted(low_risk, key=lambda x: -x[1])[:5]])
                    if len(low_risk) > 5:
                        sectors += f" [dim]+{len(low_risk)-5} more[/dim]"
                    console.print(f"  [green]● Low[/green]        {sectors}")
                console.print()
            
            console.print("[dim]────────────────────────────────────────────────────────────────────────────────[/dim]")
        
        # Summary
        summary = data.get("summary", {})
        approved = summary.get('approved', 0)
        restricted = summary.get('restricted', 0)
        quarantined = summary.get('quarantined', 0)
        rejected = summary.get('rejected', 0)
        total = summary.get('total_models', 0)
        
        console.print()
        console.print("  [bold white]SUMMARY[/bold white]")
        console.print()
        
        if approved > 0:
            console.print(f"  [green]✓ Approved[/green]     {approved}")
        if restricted > 0:
            console.print(f"  [yellow]◐ Restricted[/yellow]   {restricted}")
        if quarantined > 0:
            console.print(f"  [orange3]◑ Quarantined[/orange3]  {quarantined}")
        if rejected > 0:
            console.print(f"  [red]✗ Rejected[/red]     {rejected}")
        
        console.print()
        console.print(f"  [dim]Total: {total} model{'s' if total > 1 else ''}[/dim]")
        console.print()
        console.print("[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")
        console.print()
        
    else:
        # Fallback without Rich
        print()
        print("STRUCTURAL BACKTEST RESULTS")
        print("=" * 50)
        print(f"File: {latest.name}")
        print(f"Time: {data['timestamp'][:19]}")
        print()
        
        summary = data.get("summary", {})
        print(f"Total Models: {summary.get('total_models', 0)}")
        print(f"APPROVED: {summary.get('approved', 0)}")
        print(f"RESTRICTED: {summary.get('restricted', 0)}")
        print(f"QUARANTINED: {summary.get('quarantined', 0)}")
        print(f"REJECTED: {summary.get('rejected', 0)}")


def main():
    parser = argparse.ArgumentParser(
        description="Structural Backtest Arena CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  data      Download historical data for the 50-ticker backtest universe
  tune      Tune backtest-specific parameters (for fairness, not optimization)
  run       Execute structural backtests and apply safety rules
  results   Show latest backtest results

NON-OPTIMIZATION CONSTITUTION:
  Financial metrics are OBSERVATIONAL ONLY.
  Decisions are based on BEHAVIORAL SAFETY, not raw performance.
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Data command
    data_parser = subparsers.add_parser("data", help="Download backtest data")
    data_parser.add_argument("--force", action="store_true", help="Force re-download")
    data_parser.add_argument("--tickers", type=str, help="Override tickers (comma-separated)")
    data_parser.add_argument("--lookback", type=int, default=5, help="Years of lookback")
    
    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Tune backtest parameters")
    tune_parser.add_argument("--models", type=str, help="Models to tune (comma-separated)")
    tune_parser.add_argument("--force", action="store_true", help="Force re-tuning")
    tune_parser.add_argument("--workers", type=int, help="Number of parallel workers (default: CPU count - 1)")
    tune_parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run structural backtest")
    run_parser.add_argument("--models", type=str, help="Models to backtest (comma-separated)")
    
    # Results command
    results_parser = subparsers.add_parser("results", help="Show latest results")
    
    args = parser.parse_args()
    
    if args.command == "data":
        cmd_data(args)
    elif args.command == "tune":
        cmd_tune(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "results":
        cmd_results(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
