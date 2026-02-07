#!/usr/bin/env python3
"""
arena_cli.py — Command-line interface for Arena Model Competition

Usage:
    python src/arena/arena_cli.py data [--force]
    python src/arena/arena_cli.py tune [--symbols AAPL,NVDA,SPY]
    python src/arena/arena_cli.py results
"""

import argparse
import sys
import os

# Add src to path
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


def cmd_data(args):
    """Download arena benchmark data."""
    from arena.arena_data import download_arena_data
    from arena.arena_config import ArenaConfig, ARENA_BENCHMARK_SYMBOLS
    
    config = ArenaConfig()
    
    if args.symbols:
        config.symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    print(f"Downloading data for {len(config.symbols)} symbols...")
    datasets = download_arena_data(config, force=args.force)
    
    print(f"\nDownloaded {len(datasets)} symbols successfully")
    for symbol, dataset in datasets.items():
        print(f"  {symbol}: {dataset.n_observations} observations")


def cmd_tune(args):
    """Run arena model competition."""
    from arena.arena_tune import run_arena_competition
    from arena.arena_config import ArenaConfig
    
    config = ArenaConfig()
    config.verbose = not args.quiet
    
    if args.symbols:
        config.symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    if args.standard_only:
        config.test_experimental = False
    
    if args.experimental_only:
        config.test_standard = False
    
    result = run_arena_competition(config)
    
    return result


def cmd_results(args):
    """Show latest arena results."""
    from pathlib import Path
    import json
    
    from rich.console import Console
    
    results_dir = Path("src/data/arena/results")
    if not results_dir.exists():
        print("No arena results found. Run 'make arena-tune' first.")
        return
    
    # Find latest result file
    result_files = sorted(results_dir.glob("arena_result_*.json"), reverse=True)
    if not result_files:
        print("No arena results found. Run 'make arena-tune' first.")
        return
    
    latest = result_files[0]
    with open(latest) as f:
        data = json.load(f)
    
    console = Console()
    console.print()
    console.print("[bold]ARENA RESULTS[/bold]")
    console.print(f"[dim]{'─' * 50}[/dim]")
    console.print(f"  File: {latest.name}")
    console.print(f"  Time: {data['timestamp'][:19]}")
    console.print()
    
    # Rankings
    console.print("[bold]RANKINGS[/bold]")
    console.print(f"[dim]{'─' * 50}[/dim]")
    for i, model in enumerate(data["rankings"].get("overall", [])[:10], 1):
        if i == 1:
            console.print(f"  [bold green]#{i}  {model}[/bold green]")
        elif i <= 3:
            console.print(f"  [green]#{i}  {model}[/green]")
        else:
            console.print(f"  #{i}  {model}")
    
    # Promotion candidates
    console.print()
    if data.get("promotion_candidates"):
        console.print("[bold green]PROMOTION CANDIDATES[/bold green]")
        console.print(f"[dim]{'─' * 50}[/dim]")
        for m in data["promotion_candidates"]:
            console.print(f"  [green]>[/green] {m}")
    else:
        console.print("[yellow]No promotion candidates[/yellow]")


def cmd_disabled(args):
    """Show or manage disabled models."""
    from arena.arena_config import (
        load_disabled_models,
        enable_model,
        clear_disabled_models,
    )
    from arena.experimental_models import EXPERIMENTAL_MODELS
    
    from rich.console import Console
    
    console = Console()
    disabled = load_disabled_models()
    
    if args.clear:
        count = clear_disabled_models()
        console.print(f"[green]Re-enabled {count} model(s)[/green]")
        return
    
    if args.enable:
        model_name = args.enable
        if model_name not in disabled:
            console.print(f"[yellow]Model '{model_name}' is not disabled[/yellow]")
            return
        
        enable_model(model_name)
        console.print(f"[green]Re-enabled model: {model_name}[/green]")
        return
    
    # Show disabled models
    console.print()
    if not disabled:
        console.print("[green]All experimental models enabled[/green]")
        console.print(f"[dim]{'─' * 50}[/dim]")
        for name in EXPERIMENTAL_MODELS.keys():
            console.print(f"  [green]>[/green] {name}")
        return
    
    console.print(f"[bold red]DISABLED MODELS ({len(disabled)})[/bold red]")
    console.print(f"[dim]{'─' * 60}[/dim]")
    console.print(f"[dim]{'Model':<40}{'Gap':>10}{'Date':>12}[/dim]")
    console.print(f"[dim]{'─' * 60}[/dim]")
    
    for name, info in disabled.items():
        gap = info.get('score_gap', 0) * 100
        date = info.get("disabled_at", "")[:10]
        console.print(f"[red]{name:<40}{gap:>9.1f}%{date:>12}[/red]")
    
    # Show enabled models
    enabled = [n for n in EXPERIMENTAL_MODELS.keys() if n not in disabled]
    if enabled:
        console.print()
        console.print(f"[green]ENABLED MODELS ({len(enabled)})[/green]")
        console.print(f"[dim]{'─' * 40}[/dim]")
        for name in enabled:
            console.print(f"  [green]>[/green] {name}")
    
    console.print()
    console.print("[dim]Use --enable MODEL or --clear to re-enable[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="Arena Model Competition CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s data --force          Download fresh benchmark data
    %(prog)s tune                   Run full competition
    %(prog)s tune --symbols AAPL   Test on specific symbols
    %(prog)s results               Show latest results
    %(prog)s disabled              Show disabled models
    %(prog)s disabled --enable MODEL  Re-enable a model
    %(prog)s disabled --clear      Re-enable all models
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Data command
    data_parser = subparsers.add_parser("data", help="Download benchmark data")
    data_parser.add_argument("--force", action="store_true", help="Force re-download")
    data_parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    
    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Run model competition")
    tune_parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    tune_parser.add_argument("--standard-only", action="store_true", help="Only test standard models")
    tune_parser.add_argument("--experimental-only", action="store_true", help="Only test experimental models")
    tune_parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    
    # Results command
    results_parser = subparsers.add_parser("results", help="Show latest results")
    
    # Disabled command
    disabled_parser = subparsers.add_parser("disabled", help="Manage disabled models")
    disabled_parser.add_argument("--enable", type=str, metavar="MODEL", help="Re-enable a specific model")
    disabled_parser.add_argument("--clear", action="store_true", help="Re-enable all disabled models")
    
    args = parser.parse_args()
    
    if args.command == "data":
        cmd_data(args)
    elif args.command == "tune":
        cmd_tune(args)
    elif args.command == "results":
        cmd_results(args)
    elif args.command == "disabled":
        cmd_disabled(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
