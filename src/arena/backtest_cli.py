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
    
    # Tune
    results = tune_backtest_params(
        model_names=model_names,
        data_bundle=data_bundle,
        config=config,
        force=args.force,
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
    
    print("Running Structural Backtest Arena...")
    print(f"Models: {', '.join(model_names)}")
    print("Note: Financial metrics are OBSERVATIONAL ONLY.")
    print("      Decisions based on BEHAVIORAL SAFETY, not performance.")
    print()
    
    try:
        results = run_backtest_arena(
            model_names=model_names,
            config=config,
        )
        
        print()
        print(f"Completed backtest for {len(results)} models")
        
    except ValueError as e:
        print(f"Error: {e}")
        return


def cmd_results(args):
    """Show latest backtest results."""
    from pathlib import Path
    import json
    
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
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
        console.print()
        console.print(Panel.fit(
            "[bold]STRUCTURAL BACKTEST RESULTS[/bold]\n"
            f"[dim]File: {latest.name}[/dim]\n"
            f"[dim]Time: {data['timestamp'][:19]}[/dim]",
            border_style="cyan"
        ))
        
        # Summary
        summary = data.get("summary", {})
        console.print()
        console.print(f"  Total Models: {summary.get('total_models', 0)}")
        console.print(f"  [green]APPROVED: {summary.get('approved', 0)}[/green]")
        console.print(f"  [yellow]RESTRICTED: {summary.get('restricted', 0)}[/yellow]")
        console.print(f"  [orange3]QUARANTINED: {summary.get('quarantined', 0)}[/orange3]")
        console.print(f"  [red]REJECTED: {summary.get('rejected', 0)}[/red]")
        
        # Model details
        console.print()
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
        table.add_column("Model", style="white")
        table.add_column("Decision", justify="center")
        table.add_column("Max DD", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("Regime Stab.", justify="right")
        
        decision_colors = {
            "APPROVED": "green",
            "RESTRICTED": "yellow",
            "QUARANTINED": "orange3",
            "REJECTED": "red",
        }
        
        for model_name, model_data in data.get("models", {}).items():
            decision = model_data.get("decision", "UNKNOWN")
            color = decision_colors.get(decision, "white")
            
            fin = model_data.get("aggregate_financial", {})
            beh = model_data.get("aggregate_behavioral", {})
            
            table.add_row(
                model_name,
                f"[{color}]{decision}[/{color}]",
                f"{fin.get('max_drawdown', 0):.1%}",
                f"{fin.get('sharpe_ratio', 0):.2f}",
                f"{beh.get('regime_stability', 0):.2f}",
            )
        
        console.print(table)
        
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
