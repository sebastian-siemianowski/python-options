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
    from rich.table import Table
    from rich import box
    
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
    console.print(f"\n[bold]Latest Arena Results[/bold] ({latest.name})")
    console.print(f"Timestamp: {data['timestamp']}\n")
    
    # Rankings table
    table = Table(title="Overall Rankings", box=box.ROUNDED)
    table.add_column("Rank", justify="right")
    table.add_column("Model")
    
    for i, model in enumerate(data["rankings"].get("overall", [])[:10], 1):
        table.add_row(f"#{i}", model)
    
    console.print(table)
    
    # Promotion candidates
    if data.get("promotion_candidates"):
        console.print("\n[bold green]Promotion Candidates:[/bold green]")
        for m in data["promotion_candidates"]:
            console.print(f"  • {m}")
    else:
        console.print("\n[yellow]No promotion candidates[/yellow]")


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
    
    args = parser.parse_args()
    
    if args.command == "data":
        cmd_data(args)
    elif args.command == "tune":
        cmd_tune(args)
    elif args.command == "results":
        cmd_results(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
