#!/usr/bin/env python3
"""
refresh_data.py

Refresh price data by:
1. Deleting the last N days from all cached price files
2. Bulk downloading ALL symbols N times (always runs all passes)

This ensures fresh data. The download is unreliable so we always run multiple passes.

Usage:
    python src/data_ops/refresh_data.py
    python src/data_ops/refresh_data.py --days 5 --retries 5
    python src/data_ops/refresh_data.py --days 3 --retries 3 --workers 2
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
from rich.rule import Rule
from rich.align import Align
from rich.padding import Padding

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.data_utils import (
    DEFAULT_ASSET_UNIVERSE,
    download_prices_bulk,
    PRICE_CACHE_DIR_PATH,
    reset_symbol_tables,
    suppress_symbol_tables,
    INVERSE_CURRENCY_PAIRS,
)


def create_stocks_console() -> Console:
    """Create a beautiful console for stocks refresh."""
    return Console(force_terminal=True, width=180)


# Global console for rich output
console = create_stocks_console()


def render_stocks_header(symbols_count: int, passes: int, workers: int, batch_size: int) -> None:
    """Render extraordinary Apple-quality header for stocks refresh.
    
    Design: Cinematic title, clean stats, breathing room
    """
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # CINEMATIC TITLE
    # ═══════════════════════════════════════════════════════════════════════════════
    title = Text()
    title.append("◆", style="bold bright_cyan")
    title.append("  P R I C E   R E F R E S H", style="bold bright_white")
    console.print(Align.center(title))
    
    subtitle = Text()
    subtitle.append("Multi-Pass Bulk Download Engine", style="dim")
    console.print(Align.center(subtitle))
    console.print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # CONFIGURATION - Clean horizontal stats
    # ═══════════════════════════════════════════════════════════════════════════════
    now = datetime.now()
    
    stats = Table.grid(padding=(0, 4))
    stats.add_column(justify="center")
    stats.add_column(justify="center")
    stats.add_column(justify="center")
    stats.add_column(justify="center")
    
    stats.add_row(
        Text.assemble((f"{symbols_count:,}", "bold cyan"), ("\nsymbols", "dim")),
        Text.assemble((f"{passes}", "bold white"), ("\npasses", "dim")),
        Text.assemble((f"{workers}", "bold white"), ("\nworkers", "dim")),
        Text.assemble((now.strftime("%H:%M"), "bold white"), ("\n" + now.strftime("%b %d"), "dim")),
    )
    console.print(Align.center(stats))
    console.print()


def render_phase_header(phase_name: str, phase_num: int = None, total_phases: int = None) -> None:
    """Render a beautiful section header with numbered badge."""
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    
    header = Text()
    if phase_num:
        header.append(f"  {phase_num}  ", style="bold bright_white on blue")
        header.append("  ", style="")
    header.append(phase_name.upper(), style="bold white")
    console.print(header)
    console.print()


def trim_last_n_days(days: int = 5, quiet: bool = False) -> int:
    """Remove the last N days of data from all cached CSV files."""
    if not PRICE_CACHE_DIR_PATH.exists():
        if not quiet:
            console.print(f"    [dim]Cache directory does not exist[/dim]")
        return 0
    
    cutoff_date = datetime.now().date() - timedelta(days=days)
    files_modified = 0
    
    csv_files = list(PRICE_CACHE_DIR_PATH.glob("*.csv"))
    
    if not quiet:
        render_phase_header("TRIM CACHE")
        
        info = Text()
        info.append("    ")
        info.append(f"{len(csv_files)}", style="bold cyan")
        info.append(" files", style="dim")
        info.append("   ·   ", style="dim")
        info.append(f"{days}", style="bold white")
        info.append(" days to remove", style="dim")
        info.append("   ·   ", style="dim")
        info.append(f"cutoff {cutoff_date}", style="dim")
        console.print(info)
        console.print()
    
    if not quiet and csv_files:
        with Progress(
            SpinnerColumn(spinner_name="dots", style="bright_cyan"),
            BarColumn(bar_width=40, complete_style="cyan", finished_style="green"),
            TextColumn("[bold]{task.percentage:>5.1f}%[/bold]"),
            TextColumn("[dim]·[/dim]"),
            MofNCompleteColumn(),
            TextColumn("[dim]·[/dim]"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("", total=len(csv_files))
            
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
        
        # Show completion
        result = Text()
        result.append("    ")
        result.append("✓ ", style="green")
        result.append(f"{files_modified}", style="bold green")
        result.append(" files trimmed", style="dim")
        console.print(result)
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
    """Track download progress with Apple-quality display."""
    
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
    suppress_symbol_tables(True)  # Suppress verbose symbol tables for clean UX
    
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
    
    if not quiet:
        render_phase_header("DOWNLOAD", 2)
        
        # Stats line
        stats = Text()
        stats.append("    ")
        stats.append(f"{len(all_symbols):,}", style="bold cyan")
        stats.append(" symbols", style="dim")
        stats.append("  ·  ", style="dim")
        stats.append(f"{num_passes}", style="bold white")
        stats.append(" passes", style="dim")
        stats.append("  ·  ", style="dim")
        stats.append(f"{workers}", style="bold white")
        stats.append(" workers", style="dim")
        console.print(stats)
        console.print()
    
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
    # FIX: Track symbols that still need to be fetched across passes
    remaining_symbols = all_symbols.copy()
    
    # Set of symbols that require individual fallback (inverse currency pairs, etc.)
    # These will always fail in bulk download so we skip to final pass if only these remain
    fallback_only_symbols = set(INVERSE_CURRENCY_PAIRS.keys())
    skip_message_printed = False  # Track if skip message was printed
    
    for pass_num in range(1, num_passes + 1):
        # If no remaining symbols, we're done early
        if not remaining_symbols:
            if not quiet:
                console.print()
                done_msg = Text()
                done_msg.append("    ")
                done_msg.append("✓ ", style="bold green")
                done_msg.append("All symbols downloaded!", style="green")
                console.print(done_msg)
            break
        
        # =====================================================================
        # FAST-FORWARD: If all remaining symbols are fallback-only, skip to final pass
        # =====================================================================
        remaining_set = set(s.upper() for s in remaining_symbols)
        all_fallback_only = remaining_set.issubset(fallback_only_symbols)
        
        if all_fallback_only and pass_num < num_passes:
            if not quiet and pass_num == 1:
                # Only print skip message once (when detected at start)
                skip_msg = Text()
                skip_msg.append("    ")
                skip_msg.append("⚡ ", style="bold yellow")
                skip_msg.append(f"Skipping to final pass", style="yellow")
                skip_msg.append(f" ({len(remaining_symbols)} fallback-only symbols)", style="dim")
                console.print(skip_msg)
            # Continue to next iteration until we reach the final pass
            continue
        
        is_final_pass = (pass_num == num_passes)
        tracker.start_pass(pass_num)
        
        if not quiet:
            console.print()
            # Clean pass indicator
            pass_header = Text()
            pass_header.append("    ")
            pass_header.append(f"Pass {pass_num}", style="bold cyan")
            pass_header.append(f"/{num_passes}", style="dim")
            pass_header.append(f"  ({len(remaining_symbols)} symbols)", style="dim")
            if is_final_pass:
                pass_header.append("  +fallback", style="yellow")
            console.print(pass_header)
        
        # =====================================================================
        # INNER RETRY LOOP: Keep retrying pending symbols until all are done
        # or we've exhausted retries within this pass (max 3 inner retries)
        # =====================================================================
        # FIX: Use remaining_symbols from previous pass, not all_symbols
        symbols_to_try = remaining_symbols.copy()
        max_inner_retries = 3
        inner_retry = 0
        
        while symbols_to_try and inner_retry < max_inner_retries:
            inner_retry += 1
            
            if not quiet and inner_retry > 1:
                retry_header = Text()
                retry_header.append("      ")
                retry_header.append(f"retry {inner_retry}/{max_inner_retries}", style="dim yellow")
                retry_header.append(f"  ({len(symbols_to_try)} pending)", style="dim")
                console.print(retry_header)
            
            try:
                # Create progress bar for this attempt
                if not quiet:
                    progress_ctx = Progress(
                        SpinnerColumn(spinner_name="dots", style="cyan"),
                        BarColumn(bar_width=40, complete_style="cyan", finished_style="green"),
                        TextColumn("[bold]{task.percentage:>5.1f}%[/bold]"),
                        TextColumn("[dim]·[/dim]"),
                        MofNCompleteColumn(),
                        TextColumn("[dim]·[/dim]"),
                        TimeElapsedColumn(),
                        console=console,
                        transient=True,
                    )
                    progress_ctx.start()
                    progress_task = progress_ctx.add_task("", total=len(symbols_to_try))
                
                # Skip individual fallback on all passes except the last one
                # force_online=True ensures OFFLINE_MODE is ignored for make data
                results = download_prices_bulk(
                    symbols=symbols_to_try,
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
                
                # Find symbols that still failed
                still_pending = []
                for sym in symbols_to_try:
                    if sym not in results or results.get(sym) is None or len(results.get(sym, [])) == 0:
                        still_pending.append(sym)
                
                # Update symbols_to_try for next inner retry
                symbols_to_try = still_pending
                
                if not quiet:
                    successful_this_attempt = len(all_symbols) - len(still_pending)
                    result = Text()
                    result.append("    ")
                    if not still_pending:
                        result.append("✓ ", style="green")
                        result.append(f"{successful_this_attempt}/{len(all_symbols)}", style="bold green")
                        result.append(" complete", style="dim")
                    else:
                        result.append("● ", style="yellow")
                        result.append(f"{successful_this_attempt}", style="green")
                        result.append(" ok", style="dim")
                        result.append("  ·  ", style="dim")
                        result.append(f"{len(still_pending)}", style="yellow")
                        result.append(" pending", style="dim")
                    console.print(result)
                    
                    # Show which symbols are pending (up to 10, then summarize)
                    if still_pending and inner_retry < max_inner_retries:
                        pending_line = Text()
                        pending_line.append("      ", style="")
                        pending_line.append("→ ", style="dim yellow")
                        if len(still_pending) <= 10:
                            pending_line.append(", ".join(still_pending), style="yellow")
                        else:
                            shown = still_pending[:8]
                            remaining = len(still_pending) - 8
                            pending_line.append(", ".join(shown), style="yellow")
                            pending_line.append(f" +{remaining} more", style="dim yellow")
                        console.print(pending_line)
                
                # If no more pending, we're done with this pass
                if not still_pending:
                    break
                
                # Brief pause before inner retry
                if inner_retry < max_inner_retries:
                    time.sleep(2)
                    
            except Exception as e:
                if not quiet and progress_ctx:
                    try:
                        progress_ctx.stop()
                    except Exception:
                        pass
                if not quiet:
                    console.print(f"    [red]✗ Error:[/red] [dim]{e}[/dim]")
                # Continue to next inner retry
                time.sleep(2)
        
        # End of inner retry loop - record results for this pass
        failed_symbols = symbols_to_try  # Whatever is still pending after all inner retries
        successful = len(all_symbols) - len(failed_symbols)
        last_failed_count = len(failed_symbols)
        last_failed_symbols = failed_symbols
        
        # FIX: Update remaining_symbols for the next pass
        remaining_symbols = failed_symbols
        
        tracker.end_pass(successful, last_failed_count)
        
        # Check if we're done
        if last_failed_count == 0:
            if not quiet:
                console.print()
                done_msg = Text()
                done_msg.append("    ")
                done_msg.append("✓ ", style="bold green")
                done_msg.append("All symbols downloaded!", style="green")
                console.print(done_msg)
            break
        
        # Show final pending list for this pass
        if not quiet and last_failed_symbols:
            pending_line = Text()
            pending_line.append("      ", style="")
            pending_line.append("→ ", style="dim yellow")
            if len(last_failed_symbols) <= 10:
                pending_line.append(", ".join(last_failed_symbols), style="yellow")
            else:
                shown = last_failed_symbols[:8]
                remaining = len(last_failed_symbols) - 8
                pending_line.append(", ".join(shown), style="yellow")
                pending_line.append(f" +{remaining} more", style="dim yellow")
            console.print(pending_line)

        # Wait between passes (with countdown)
        if pass_num < num_passes:
            if not quiet:
                wait_time = 5
                with Progress(
                    SpinnerColumn(spinner_name="dots", style="dim"),
                    TextColumn("[dim]Next pass in[/dim]"),
                    TextColumn("[cyan]{task.completed}s[/cyan]"),
                    console=console,
                    transient=True,
                ) as progress:
                    task = progress.add_task("", total=wait_time)
                    for i in range(wait_time):
                        time.sleep(1)
                        progress.update(task, completed=wait_time - i - 1)
            else:
                time.sleep(5)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY - Apple-quality
    # ═══════════════════════════════════════════════════════════════════════════════
    if not quiet:
        console.print()
        render_phase_header("SUMMARY", 3)
        
        # Pass history as clean table
        table = Table(
            show_header=True,
            header_style="dim",
            box=None,
            padding=(0, 2),
        )
        table.add_column("Pass", width=8)
        table.add_column("Success", justify="right", width=8)
        table.add_column("Pending", justify="right", width=8)
        table.add_column("", width=10)
        
        for i, (s, f) in enumerate(tracker.pass_results, 1):
            if f == 0:
                status = "[green]✓[/green]"
            else:
                status = f"[yellow]{f} left[/yellow]"
            table.add_row(
                f"[bold]{i}[/bold]",
                f"[green]{s}[/green]",
                f"[yellow]{f}[/yellow]" if f > 0 else "[dim]—[/dim]",
                status
            )
        
        console.print(Padding(table, (0, 0, 0, 4)))
        
        # Show failed symbols if any
        if last_failed_count > 0 and last_failed_symbols:
            console.print()
            warning = Text()
            warning.append("    ")
            warning.append("⚠ ", style="yellow")
            warning.append(f"{last_failed_count}", style="bold yellow")
            warning.append(" symbols may have issues", style="dim")
            console.print(warning)
            
            # Show first 8 in compact format
            shown = last_failed_symbols[:8]
            console.print(f"      [dim]{', '.join(shown)}[/dim]")
            if len(last_failed_symbols) > 8:
                console.print(f"      [dim]...and {len(last_failed_symbols) - 8} more[/dim]")
    
    return last_failed_count


def render_complete_banner(total: int, failed: int) -> None:
    """Render a beautiful Apple-quality completion banner."""
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    
    if failed == 0:
        # Success state - elegant panel
        content = Text()
        content.append("\n", style="")
        content.append("✓", style="bold bright_green")
        content.append("  ", style="")
        content.append("Complete", style="bold white")
        content.append("\n\n", style="")
        content.append(f"{total:,}", style="bold bright_green")
        content.append(" symbols refreshed", style="dim")
        content.append("\n", style="")
        
        success_panel = Panel(
            Align.center(content),
            box=box.ROUNDED,
            border_style="bright_green",
            padding=(0, 4),
        )
        console.print(Align.center(success_panel, width=50))
    else:
        # Warning state
        content = Text()
        content.append("\n", style="")
        content.append("⚠", style="bold bright_yellow")
        content.append("  ", style="")
        content.append("Complete with warnings", style="bold white")
        content.append("\n\n", style="")
        content.append(f"{total - failed:,}", style="bold bright_green")
        content.append(" loaded", style="dim")
        content.append("  ·  ", style="dim")
        content.append(f"{failed:,}", style="bold bright_yellow")
        content.append(" issues", style="dim")
        content.append("\n", style="")
        
        warning_panel = Panel(
            Align.center(content),
            box=box.ROUNDED,
            border_style="bright_yellow",
            padding=(0, 4),
        )
        console.print(Align.center(warning_panel, width=50))
    
    console.print()


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

    # Get all symbols first for header
    all_symbols = get_all_symbols()
    
    if not args.quiet:
        render_stocks_header(
            symbols_count=len(all_symbols),
            passes=args.retries,
            workers=args.workers,
            batch_size=args.batch_size
        )

    # Trim last N days from cache (unless skipped)
    if not args.skip_trim:
        trim_last_n_days(days=args.days, quiet=args.quiet)
    else:
        if not args.quiet:
            render_phase_header("CACHE TRIM", 1)
            console.print("    [dim]Skipped (--skip-trim)[/dim]")

    # Bulk download N times (always runs all passes)
    failed_count = bulk_download_n_times(
        symbols=all_symbols,
        num_passes=args.retries,
        batch_size=args.batch_size,
        workers=args.workers,
        years=args.years,
        quiet=args.quiet,
    )

    if not args.quiet:
        render_complete_banner(len(all_symbols), failed_count)
        
        # Show Metals Risk Temperature after data refresh
        try:
            from decision.metals_risk_temperature import (
                compute_metals_risk_temperature,
                render_metals_risk_temperature,
            )
            metals_result = compute_metals_risk_temperature(start_date="2020-01-01")
            render_metals_risk_temperature(metals_result, console=console)
        except Exception as e:
            # Silently skip if metals temperature computation fails
            if os.getenv('DEBUG'):
                console.print(f"[dim]Metals temperature unavailable: {e}[/dim]")

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
