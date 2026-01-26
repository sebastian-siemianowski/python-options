#!/usr/bin/env python3
"""
refresh_data.py

Refresh price data by:
1. Deleting the last N days from all cached price files
2. Bulk downloading ALL symbols N times (always runs all passes)

This ensures fresh data. The download is unreliable so we always run multiple passes.

Usage:
    python scripts/refresh_data.py
    python scripts/refresh_data.py --days 5 --retries 5
    python scripts/refresh_data.py --days 3 --retries 3 --workers 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

# Rich imports for beautiful UX
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich import box

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fx_data_utils import (
    DEFAULT_ASSET_UNIVERSE,
    download_prices_bulk,
    PRICE_CACHE_DIR_PATH,
    reset_symbol_tables,
)

# Global console for rich output
console = Console()


def create_header_panel(days: int, retries: int, batch_size: int, workers: int) -> Panel:
    """Create a beautiful header panel with configuration info."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("📁 Cache", str(PRICE_CACHE_DIR_PATH))
    table.add_row("📅 Days to trim", str(days))
    table.add_row("🔄 Download passes", f"{retries} (always runs all)")
    table.add_row("📦 Batch size", str(batch_size))
    table.add_row("👷 Workers", str(workers))
    
    return Panel(
        table,
        title="[bold cyan]📊 Price Data Refresh[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )


def trim_last_n_days(days: int = 5, quiet: bool = False) -> int:
    """Remove the last N days of data from all cached CSV files."""
    if not PRICE_CACHE_DIR_PATH.exists():
        if not quiet:
            console.print(f"[yellow]Cache directory does not exist:[/yellow] {PRICE_CACHE_DIR_PATH}")
        return 0
    
    cutoff_date = datetime.now().date() - timedelta(days=days)
    files_modified = 0
    
    csv_files = list(PRICE_CACHE_DIR_PATH.glob("*.csv"))
    
    if not quiet:
        console.print()
        console.print(f"[bold]Step 1:[/bold] Trimming last [cyan]{days}[/cyan] days from [cyan]{len(csv_files)}[/cyan] cache files...")
        console.print(f"[dim]Cutoff date: {cutoff_date}[/dim]")
    
    if not quiet and csv_files:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Trimming files...", total=len(csv_files))
            
            for csv_path in csv_files:
                try:
                    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    if df.empty:
                        progress.advance(task)
                        continue
                    
                    original_len = len(df)
                    df_trimmed = df[df.index.date < cutoff_date]
                    
                    if len(df_trimmed) < original_len:
                        df_trimmed.to_csv(csv_path)
                        files_modified += 1
                        
                except Exception:
                    pass
                
                progress.advance(task)
        
        console.print(f"  [green]✓[/green] Modified [bold]{files_modified}[/bold] files")
    else:
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if df.empty:
                    continue
                
                original_len = len(df)
                df_trimmed = df[df.index.date < cutoff_date]
                
                if len(df_trimmed) < original_len:
                    df_trimmed.to_csv(csv_path)
                    files_modified += 1
                    
            except Exception:
                pass
    
    return files_modified


def get_all_symbols() -> List[str]:
    """Get all symbols that should be downloaded (universe + FX pairs)."""
    all_symbols = list(DEFAULT_ASSET_UNIVERSE)
    
    fx_pairs = [
        "USDPLN=X", "EURPLN=X", "GBPPLN=X", "PLNJPY=X",
        "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X",
        "AUDUSD=X", "USDCAD=X", "CHFPLN=X", "CADPLN=X",
        "AUDPLN=X", "SEKPLN=X",
    ]
    
    return list(dict.fromkeys(all_symbols + fx_pairs))


class DownloadProgressTracker:
    """Track download progress for Rich live display."""
    
    def __init__(self, total_symbols: int, num_passes: int):
        self.total_symbols = total_symbols
        self.num_passes = num_passes
        self.current_pass = 0
        self.cached_count = 0
        self.total_need = 0
        self.fetched_count = 0
        self.from_cache = 0
        self.chunks_total = 0
        self.last_successful = 0
        self.last_failed = 0
        self.pass_results: List[tuple] = []  # (successful, failed) per pass
        
    def start_pass(self, pass_num: int):
        self.current_pass = pass_num
        self.fetched_count = 0
        
    def update_bulk_info(self, uncached: int, from_cache: int, chunks: int):
        self.total_need = uncached
        self.from_cache = from_cache
        self.chunks_total = chunks
        
    def update_progress(self, fetched: int):
        self.fetched_count = fetched
        
    def end_pass(self, successful: int, failed: int):
        self.last_successful = successful
        self.last_failed = failed
        self.pass_results.append((successful, failed))
        
    def create_display(self) -> Panel:
        """Create a rich panel showing current progress."""
        # Main progress table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("", width=25)
        table.add_column("", width=50)
        
        # Pass progress bar
        pass_pct = (self.current_pass / self.num_passes) * 100 if self.num_passes > 0 else 0
        pass_bar = self._make_bar(pass_pct, 30, "cyan")
        table.add_row(
            f"[bold]Pass Progress[/bold]",
            f"{pass_bar} [cyan]{self.current_pass}/{self.num_passes}[/cyan]"
        )
        
        # Download progress bar (within current pass)
        if self.total_need > 0:
            dl_pct = (self.fetched_count / self.total_need) * 100
            dl_bar = self._make_bar(dl_pct, 30, "green")
            table.add_row(
                f"[bold]Download Progress[/bold]",
                f"{dl_bar} [green]{self.fetched_count}/{self.total_need}[/green]"
            )
        
        # Stats
        table.add_row("", "")
        table.add_row("[dim]From cache:[/dim]", f"[yellow]{self.from_cache}[/yellow]")
        table.add_row("[dim]Chunks:[/dim]", f"[yellow]{self.chunks_total}[/yellow]")
        
        # Pass history
        if self.pass_results:
            table.add_row("", "")
            history_parts = []
            for i, (s, f) in enumerate(self.pass_results, 1):
                if f == 0:
                    history_parts.append(f"[green]P{i}:✓{s}[/green]")
                else:
                    history_parts.append(f"[yellow]P{i}:{s}✓/{f}✗[/yellow]")
            table.add_row("[dim]History:[/dim]", " ".join(history_parts))
        
        is_final = self.current_pass == self.num_passes
        title_suffix = " [yellow](final - with fallback)[/yellow]" if is_final else ""
        
        return Panel(
            table,
            title=f"[bold cyan]🔄 Pass {self.current_pass}/{self.num_passes}[/bold cyan]{title_suffix}",
            border_style="cyan",
        )
    
    def _make_bar(self, pct: float, width: int, color: str) -> str:
        """Create a text-based progress bar."""
        filled = int((pct / 100) * width)
        empty = width - filled
        return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"


def bulk_download_n_times(
    symbols: List[str],
    num_passes: int = 5,
    batch_size: int = 16,
    workers: int = 12,
    years: int = 10,
    quiet: bool = False,
) -> int:
    """
    Download data for symbols, ALWAYS running num_passes times with BULK ONLY.
    Individual fallback only happens on the final pass for any remaining symbols.
    Each pass downloads ALL symbols to ensure reliability.
    """
    # Reset symbol tables to prevent duplicates from previous runs
    reset_symbol_tables()
    
    if not symbols:
        if not quiet:
            console.print("[yellow]No symbols to download.[/yellow]")
        return 0
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=years * 365)
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()
    
    all_symbols = list(symbols)
    last_failed_count = len(all_symbols)
    last_failed_symbols = all_symbols.copy()
    
    # Create progress tracker
    tracker = DownloadProgressTracker(len(all_symbols), num_passes)
    
    # For live progress display - declared here so rich_log can access them
    progress_ctx: Optional[Progress] = None
    progress_task = None
    
    # Custom log function that updates progress bar
    def rich_log(msg: str):
        nonlocal progress_ctx, progress_task
        # Parse the log message to extract progress info
        if "Bulk download:" in msg:
            # Parse: "Bulk download: X uncached, Y from cache, Z chunk(s)"
            try:
                parts = msg.split(",")
                uncached = int(parts[0].split(":")[1].strip().split()[0])
                from_cache = int(parts[1].strip().split()[0])
                chunks = int(parts[2].strip().split()[0])
                tracker.update_bulk_info(uncached, from_cache, chunks)
                # Update progress bar description and total
                if progress_ctx is not None and progress_task is not None:
                    if uncached > 0:
                        progress_ctx.update(progress_task, total=uncached, completed=0,
                                           description=f"[cyan]Downloading ({from_cache} cached, {chunks} chunks, {workers} workers)...[/cyan]")
                    else:
                        progress_ctx.update(progress_task, total=from_cache, completed=from_cache,
                                           description=f"[green]All {from_cache} symbols from cache![/green]")
            except Exception:
                pass
        elif "Cached" in msg and "/" in msg:
            # Parse: "✓ Cached X/Y (Z%) so far"
            try:
                # Extract X from "Cached X/Y"
                cached_part = msg.split("Cached")[1].strip()
                fetched = int(cached_part.split("/")[0])
                tracker.update_progress(fetched)
                # Update progress bar
                if progress_ctx is not None and progress_task is not None:
                    progress_ctx.update(progress_task, completed=fetched)
            except Exception:
                pass
    
    # Run num_passes bulk-only passes (no individual fallback)
    for pass_num in range(1, num_passes + 1):
        is_final_pass = (pass_num == num_passes)
        tracker.start_pass(pass_num)
        
        if not quiet:
            console.print()
            is_final_str = " [yellow](final - with individual fallback)[/yellow]" if is_final_pass else " [dim](bulk only)[/dim]"
            console.print(Panel(
                f"Processing [cyan]{len(all_symbols)}[/cyan] symbols (cached will be skipped)...",
                title=f"[bold]🔄 Pass {pass_num}/{num_passes}[/bold]{is_final_str}",
                border_style="blue",
            ))
        
        try:
            # Create progress bar for this pass
            if not quiet:
                progress_ctx = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=40),
                    TextColumn("[green]{task.completed}/{task.total}[/green]"),
                    TextColumn("[cyan]({task.percentage:>5.1f}%)[/cyan]"),
                    TimeElapsedColumn(),
                    console=console,
                    transient=False,
                )
                progress_ctx.start()
                progress_task = progress_ctx.add_task("[cyan]Checking cache...[/cyan]", total=len(all_symbols))
            
            # Skip individual fallback on all passes except the last one
            # force_online=True ensures OFFLINE_MODE is ignored for make data
            results = download_prices_bulk(
                symbols=all_symbols,
                start=start_str,
                end=end_str,
                chunk_size=batch_size,
                progress=not quiet,
                log_fn=rich_log if not quiet else None,
                skip_individual_fallback=not is_final_pass,
                max_workers=workers,
                force_online=True,  # Always try to download for make data
            )
            
            if not quiet and progress_ctx is not None:
                progress_ctx.stop()
            
            failed_symbols = []
            for sym in all_symbols:
                if sym not in results or results.get(sym) is None or len(results.get(sym, [])) == 0:
                    failed_symbols.append(sym)
            
            successful = len(all_symbols) - len(failed_symbols)
            last_failed_count = len(failed_symbols)
            last_failed_symbols = failed_symbols
            
            tracker.end_pass(successful, last_failed_count)
            
            if not quiet:
                console.print()  # Line break after progress bar
                # Show pass results
                if last_failed_count == 0:
                    console.print(f"  [green]✓[/green] Pass {pass_num}: [green]{successful}/{len(all_symbols)} successful[/green] — [bold green]All complete![/bold green]")
                else:
                    console.print(f"  [yellow]⚡[/yellow] Pass {pass_num}: [green]{successful}[/green] successful, [yellow]{last_failed_count}[/yellow] pending")
            
            # Wait between passes (with countdown)
            if pass_num < num_passes:
                if not quiet:
                    wait_time = 5
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[dim]Waiting before next pass...[/dim]"),
                        BarColumn(bar_width=20),
                        TextColumn("[cyan]{task.completed}s/{task.total}s[/cyan]"),
                        console=console,
                        transient=True,
                    ) as progress:
                        task = progress.add_task("", total=wait_time)
                        for i in range(wait_time):
                            time.sleep(1)
                            progress.update(task, completed=i + 1)
                else:
                    time.sleep(5)
                
        except Exception as e:
            # Stop progress bar on error
            if not quiet and progress_ctx:
                try:
                    progress_ctx.stop()
                except Exception:
                    pass
            if not quiet:
                console.print(f"[red]Error during bulk download pass {pass_num}:[/red] {e}")
            tracker.end_pass(0, len(all_symbols))
            if pass_num < num_passes:
                time.sleep(5)
    
    # Final summary
    if not quiet:
        console.print()
        
        # Create summary table
        summary_table = Table(
            title="📊 Download Summary",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        summary_table.add_column("Pass", justify="center", width=8)
        summary_table.add_column("Successful", justify="right", style="green", width=12)
        summary_table.add_column("Failed", justify="right", style="yellow", width=12)
        summary_table.add_column("Status", justify="center", width=15)
        
        for i, (s, f) in enumerate(tracker.pass_results, 1):
            status = "[green]✓ Complete[/green]" if f == 0 else f"[yellow]{f} pending[/yellow]"
            summary_table.add_row(f"Pass {i}", str(s), str(f), status)
        
        # Add total row
        total_successful = tracker.pass_results[-1][0] if tracker.pass_results else 0
        total_failed = tracker.pass_results[-1][0] if tracker.pass_results else len(all_symbols)
        summary_table.add_row(
            "[bold]Final[/bold]",
            f"[bold]{total_successful}[/bold]",
            f"[bold]{last_failed_count}[/bold]",
            "[bold green]✓ Done[/bold green]" if last_failed_count == 0 else f"[bold yellow]{last_failed_count} issues[/bold yellow]",
            style="bold",
        )
        
        console.print(summary_table)
        
        if last_failed_count > 0 and last_failed_symbols:
            console.print()
            console.print(f"[yellow]⚠️  {last_failed_count} symbols may have issues:[/yellow]")
            # Show first 10 in a compact format
            shown = last_failed_symbols[:10]
            console.print(f"  [dim]{', '.join(shown)}[/dim]")
            if len(last_failed_symbols) > 10:
                console.print(f"  [dim]... and {len(last_failed_symbols) - 10} more[/dim]")
    
    return last_failed_count


def main():
    parser = argparse.ArgumentParser(
        description="Refresh price data by trimming recent days and re-downloading"
    )
    parser.add_argument("--days", type=int, default=5, help="Days to trim from cache (default: 5)")
    parser.add_argument("--retries", type=int, default=5, help="Download passes - ALWAYS runs this many (default: 5)")
    parser.add_argument("--workers", type=int, default=2, help="Parallel download workers (default: 2)")
    parser.add_argument("--batch-size", type=int, default=16, help="Symbols per batch (default: 16)")
    parser.add_argument("--years", type=int, default=10, help="Years of history (default: 10)")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--skip-trim", action="store_true", help="Skip trimming, only download")

    args = parser.parse_args()

    if not args.quiet:
        console.print()
        console.print(create_header_panel(args.days, args.retries, args.batch_size, args.workers))

    # Step 1: Trim last N days from cache
    if not args.skip_trim:
        trim_last_n_days(days=args.days, quiet=args.quiet)
    else:
        if not args.quiet:
            console.print()
            console.print(f"[bold]Step 1:[/bold] [dim]Skipping trim (--skip-trim)[/dim]")

    # Step 2: Get all symbols
    all_symbols = get_all_symbols()
    if not args.quiet:
        console.print()
        console.print(f"[bold]Step 2:[/bold] Found [cyan]{len(all_symbols)}[/cyan] symbols to download")

    # Step 3: Bulk download N times (always runs all passes)
    if not args.quiet:
        console.print()
        console.print(f"[bold]Step 3:[/bold] Running [cyan]{args.retries}[/cyan] bulk download passes...")
    
    failed_count = bulk_download_n_times(
        symbols=all_symbols,
        num_passes=args.retries,
        batch_size=args.batch_size,
        workers=args.workers,
        years=args.years,
        quiet=args.quiet,
    )

    if not args.quiet:
        console.print()
        if failed_count == 0:
            console.print(Panel(
                f"[green]All [bold]{len(all_symbols)}[/bold] symbols downloaded successfully![/green]",
                title="[bold green]✅ Refresh Complete[/bold green]",
                border_style="green",
            ))
        else:
            console.print(Panel(
                f"Completed {args.retries} passes for {len(all_symbols)} symbols\n"
                f"[yellow]{failed_count} symbols may have issues[/yellow]",
                title="[bold yellow]⚠️  Refresh Complete (with warnings)[/bold yellow]",
                border_style="yellow",
            ))

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
