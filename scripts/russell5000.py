#!/usr/bin/env python3
"""
Builds a Russell 5000 tickers CSV at data/universes/russell5000_tickers.csv by scraping
public sources and ranking all US listed stocks by market cap to select 5,000 names.

Features:
- Uses multiprocessing (all CPU cores) for maximum throughput
- Beautiful Rich library UX with progress bars and panels
- Robust market cap fetching with multiple fallbacks

Usage:
  make russell5000
  # or directly:
  python scripts/russell5000.py [--out data/universes/russell5000_tickers.csv]
"""
import argparse
import os
import sys
import time
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count, Manager
from datetime import datetime

import pandas as pd
import numpy as np

# Rich library for beautiful UX
try:
    from rich.console import Console, Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn, MofNCompleteColumn, TimeRemainingColumn
    from rich.text import Text
    from rich import box
    from rich.columns import Columns
    from rich.rule import Rule
    from rich.align import Align
    from rich.live import Live
    from rich.layout import Layout
    from rich.style import Style
    from rich.padding import Padding
    from rich.spinner import Spinner
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

DEFAULT_OUT = "data/universes/russell5000_tickers.csv"
DATA_DIR = "data"

# NASDAQ Trader symbol directories (pipe-delimited)
NASDAQ_LISTED_URL = "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

TARGET_SIZE = 5000
MIN_EXPECTED_CANDIDATES = 5100  # Must be > TARGET_SIZE to allow for invalid caps

# Cache locations
META_DIR = os.path.join(DATA_DIR, "meta")
CAPS_CACHE = os.path.join(META_DIR, "caps_cache.csv")


def normalize_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    s = s.replace(" ", "")
    if "." in s:
        s = s.replace(".", "-")
    return s


def _read_nasdaq_table(url: str, symbol_col: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url, sep='|', engine='python', dtype=str)
        df = df[df[symbol_col].notna()]
        return df
    except Exception as e:
        return pd.DataFrame()


def fetch_us_primary_listings_from_nasdaq_trader() -> List[str]:
    """Fetch US primary listings from NASDAQ Trader directories."""
    dfs = []
    nasdaq = _read_nasdaq_table(NASDAQ_LISTED_URL, symbol_col='Symbol')
    if not nasdaq.empty:
        nasdaq = nasdaq.rename(columns={
            "Symbol": "SYMBOL", 
            "Security Name": "SECURITY_NAME", 
            "ETF": "ETF", 
            "Test Issue": "TEST_ISSUE"
        })
        dfs.append(nasdaq[["SYMBOL", "SECURITY_NAME", "ETF", "TEST_ISSUE"]])
    
    other = _read_nasdaq_table(OTHER_LISTED_URL, symbol_col='ACT Symbol')
    if not other.empty:
        other = other.rename(columns={
            "ACT Symbol": "SYMBOL", 
            "Security Name": "SECURITY_NAME", 
            "ETF": "ETF", 
            "Test Issue": "TEST_ISSUE"
        })
        dfs.append(other[["SYMBOL", "SECURITY_NAME", "ETF", "TEST_ISSUE"]])

    if not dfs:
        return []

    all_df = pd.concat(dfs, ignore_index=True)
    all_df["SYMBOL"] = all_df["SYMBOL"].astype(str).str.strip()
    all_df["SECURITY_NAME"] = all_df.get("SECURITY_NAME", "").astype(str)
    all_df["ETF"] = all_df.get("ETF", "N").astype(str).str.upper().fillna("N")
    all_df["TEST_ISSUE"] = all_df.get("TEST_ISSUE", "N").astype(str).str.upper().fillna("N")

    mask_keep = (
        (all_df["ETF"] != "Y") &
        (all_df["TEST_ISSUE"] != "Y")
    )

    bad_keywords = [
        "PREFERRED", "PFD", "WARRANT", "RIGHTS", "UNIT", "UNITS",
        "NOTE", "BOND", "DEPOSITARY", "ADR", "FUND", "TRUST"
    ]
    upper_names = all_df["SECURITY_NAME"].str.upper().fillna("")
    for kw in bad_keywords:
        mask_keep &= ~upper_names.str.contains(rf"\b{kw}\b", regex=True)

    filtered = all_df[mask_keep]
    syms = [normalize_symbol(s) for s in filtered["SYMBOL"].tolist()]
    
    valid_syms = []
    for s in syms:
        if not s:
            continue
        if any(ch in s for ch in [" ", "/", "^"]):
            continue
        valid_syms.append(s)

    return sorted(set(valid_syms))


def _load_caps_cache() -> Dict[str, float]:
    if not os.path.exists(CAPS_CACHE):
        return {}
    try:
        df = pd.read_csv(CAPS_CACHE)
        out = {}
        for _, row in df.iterrows():
            sym = str(row.get("ticker", "")).strip().upper()
            val = row.get("marketCap", None)
            try:
                cap = float(val) if pd.notna(val) else float("nan")
            except Exception:
                cap = float("nan")
            if sym:
                out[sym] = cap
        return out
    except Exception:
        return {}


def _save_caps_cache(cache: Dict[str, float]) -> None:
    os.makedirs(META_DIR, exist_ok=True)
    rows = [(k, v) for k, v in cache.items() if pd.notna(v) and v > 0]
    df = pd.DataFrame(rows, columns=["ticker", "marketCap"])
    df.to_csv(CAPS_CACHE, index=False)


def _fetch_cap_worker(ticker: str) -> Tuple[str, float]:
    """Worker function for multiprocessing - fetches market cap for a single ticker."""
    import yfinance as yf
    
    try:
        tk = yf.Ticker(ticker)
        
        # 1) fast_info
        try:
            fi = getattr(tk, "fast_info", None)
            if fi is not None:
                if isinstance(fi, dict):
                    m = fi.get("market_cap") or fi.get("marketCap")
                    if m is not None and float(m) > 0:
                        return (ticker, float(m))
                else:
                    v = getattr(fi, 'market_cap', None) or getattr(fi, 'marketCap', None)
                    if v is not None and float(v) > 0:
                        return (ticker, float(v))
        except Exception:
            pass
        
        # 2) info['marketCap']
        info = {}
        try:
            info = tk.info or {}
            m = info.get("marketCap") or info.get("market_cap")
            if m is not None and float(m) > 0:
                return (ticker, float(m))
        except Exception:
            info = {}
        
        # 3) sharesOutstanding * price
        price = None
        try:
            fi = getattr(tk, "fast_info", None)
            if fi is not None:
                if isinstance(fi, dict):
                    price = fi.get('last_price') or fi.get('regular_market_price') or fi.get('previous_close')
                else:
                    for attr in ("last_price", "regular_market_price", "previous_close"):
                        val = getattr(fi, attr, None)
                        if val is not None:
                            price = float(val)
                            break
            if price is None and info:
                price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
                if price:
                    price = float(price)
        except Exception:
            pass
        
        shares = None
        try:
            so = info.get('sharesOutstanding') if info else None
            if so is not None:
                shares = float(so)
        except Exception:
            pass
        
        if price and price > 0 and shares and shares > 0:
            return (ticker, float(price) * float(shares))
        
        return (ticker, float('nan'))
    except Exception:
        return (ticker, float('nan'))


def _process_batch(args: Tuple[List[str], Dict[str, float]]) -> List[Tuple[str, float]]:
    """Process a batch of tickers, using cache when available."""
    tickers, cache = args
    results = []
    for ticker in tickers:
        key = ticker.upper()
        if key in cache and pd.notna(cache[key]) and cache[key] > 0:
            results.append((ticker, cache[key]))
        else:
            results.append(_fetch_cap_worker(ticker))
    return results


def human_money(x: float) -> str:
    try:
        n = float(x)
    except Exception:
        return "-"
    sign = "-" if n < 0 else ""
    n = abs(n)
    units = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]
    for thresh, suffix in units:
        if n >= thresh:
            val = n / thresh
            return f"{sign}{val:,.1f}{suffix}"
    return f"{sign}{n:,.0f}"


def rank_by_market_cap_multiprocess(
    universe: List[str], 
    num_processes: int = 0,
    use_rich: bool = True
) -> List[Tuple[str, float]]:
    """Rank tickers by market cap using multiprocessing for maximum throughput."""
    
    cache = _load_caps_cache()
    
    # Determine number of processes
    if num_processes <= 0:
        num_processes = cpu_count()
    
    # Split into batches for multiprocessing
    batch_size = max(1, len(universe) // (num_processes * 4))  # 4 batches per process
    batches = []
    for i in range(0, len(universe), batch_size):
        batch = universe[i:i + batch_size]
        batches.append((batch, cache))
    
    results: List[Tuple[str, float]] = []
    processed_count = 0
    
    if use_rich and RICH_AVAILABLE:
        console = Console()
        
        # Apple-style minimal progress
        with Progress(
            TextColumn("    "),
            SpinnerColumn(spinner_name="dots", style="white"),
            TextColumn("[dim]{task.description}[/dim]"),
            BarColumn(bar_width=30, style="dim white", complete_style="white", finished_style="bright_white"),
            TextColumn("[bold white]{task.percentage:>3.0f}%[/bold white]"),
            TextColumn("[dim]•[/dim]"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Analyzing symbols", total=len(batches))
            
            with Pool(processes=num_processes) as pool:
                for batch_results in pool.imap_unordered(_process_batch, batches):
                    results.extend(batch_results)
                    processed_count += len(batch_results)
                    progress.advance(task)
    else:
        # Fallback without rich
        from tqdm import tqdm
        with Pool(processes=num_processes) as pool:
            for batch_results in tqdm(pool.imap_unordered(_process_batch, batches), 
                                       total=len(batches), 
                                       desc="    Analyzing",
                                       ncols=60,
                                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
                results.extend(batch_results)
    
    # Update cache with new results
    for ticker, cap in results:
        if pd.notna(cap) and cap > 0:
            cache[ticker.upper()] = cap
    _save_caps_cache(cache)
    
    # Build DataFrame and sort
    df = pd.DataFrame(results, columns=["ticker", "marketCap"])
    df = df.dropna(subset=["marketCap"])
    df = df[df["marketCap"] > 0]
    df = df.sort_values("marketCap", ascending=True).reset_index(drop=True)
    
    return list(df.itertuples(index=False, name=None))


def main():
    parser = argparse.ArgumentParser(
        description="🚀 Build Russell 5000 tickers CSV from public sources using multiprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  make russell5000                        # Build with defaults
  python scripts/russell5000.py     # Same as above
  python scripts/russell5000.py --min_size 3000  # Build smaller universe
        """
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output CSV path (default: {DEFAULT_OUT})")
    parser.add_argument("--min_size", type=int, default=TARGET_SIZE, help=f"Target list size (default: {TARGET_SIZE})")
    parser.add_argument("--processes", type=int, default=0, help="Number of processes (0=auto, use all cores)")
    parser.add_argument("--plain", action="store_true", help="Disable rich output")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)

    use_rich = RICH_AVAILABLE and not args.plain
    num_processes = args.processes if args.processes > 0 else cpu_count()
    start_time_total = time.time()
    
    if use_rich:
        console = Console()
        
        # Clear screen for immersive experience
        console.clear()
        console.print()
        
        # ═══════════════════════════════════════════════════════════════════════════
        # HERO HEADER - Stunning visual impact
        # ═══════════════════════════════════════════════════════════════════════════
        header_content = Text()
        header_content.append("\n")
        header_content.append("R U S S E L L   5 0 0 0", style="bold bright_white")
        header_content.append("\n")
        header_content.append("Universe Builder", style="dim italic")
        header_content.append("\n")
        
        console.print(Panel(
            Align.center(header_content),
            border_style="bright_white",
            padding=(1, 8),
            box=box.DOUBLE,
        ))
        console.print()
        
        # ═══════════════════════════════════════════════════════════════════════════
        # SYSTEM INFO - Clean minimal metrics
        # ═══════════════════════════════════════════════════════════════════════════
        info_grid = Table.grid(padding=(0, 6))
        info_grid.add_column(justify="center")
        info_grid.add_column(justify="center")
        info_grid.add_column(justify="center")
        info_grid.add_row(
            Text.assemble(("◉ ", "green"), (f"{num_processes}", "bold white"), (" cores", "dim")),
            Text.assemble(("◎ ", "cyan"), (f"{args.min_size:,}", "bold white"), (" target", "dim")),
            Text.assemble(("◉ ", "yellow"), (datetime.now().strftime("%H:%M"), "bold white"), (" start", "dim")),
        )
        console.print(Align.center(info_grid))
        console.print()
        console.print()
        
        # ═══════════════════════════════════════════════════════════════════════════
        # PHASE 1: DISCOVERY
        # ═══════════════════════════════════════════════════════════════════════════
        phase1_title = Text()
        phase1_title.append("  1  ", style="bold black on white")
        phase1_title.append("  DISCOVERY", style="bold white")
        console.print(phase1_title)
        console.print()
        
        # Animated status for fetching
        with console.status(
            Text.assemble(("    ", ""), ("Connecting to NASDAQ...", "dim italic")),
            spinner="dots",
            spinner_style="white"
        ):
            candidates = fetch_us_primary_listings_from_nasdaq_trader()
        
        if len(candidates) < MIN_EXPECTED_CANDIDATES:
            console.print()
            console.print(Panel(
                f"[bold red]Insufficient Data[/bold red]\n\n"
                f"Found only [bold]{len(candidates):,}[/bold] candidates.\n"
                f"Need at least [bold]{MIN_EXPECTED_CANDIDATES:,}[/bold] for a valid universe.",
                border_style="red",
                padding=(1, 4)
            ))
            sys.exit(2)
        
        # Success indicator with details
        console.print(f"    [green]●[/green]  [bold white]{len(candidates):,}[/bold white] [dim]symbols discovered[/dim]")
        console.print()
        console.print()
        
        # ═══════════════════════════════════════════════════════════════════════════
        # PHASE 2: ANALYSIS  
        # ═══════════════════════════════════════════════════════════════════════════
        phase2_title = Text()
        phase2_title.append("  2  ", style="bold black on white")
        phase2_title.append("  ANALYSIS", style="bold white")
        console.print(phase2_title)
        console.print()
        
        # Custom progress with clean design
        analysis_start = time.time()
        ranked = rank_by_market_cap_multiprocess(candidates, num_processes, use_rich=True)
        analysis_elapsed = time.time() - analysis_start
        
        # Calculate throughput
        throughput = len(candidates) / analysis_elapsed if analysis_elapsed > 0 else 0
        
        console.print()
        console.print(f"    [green]●[/green]  [bold white]{len(ranked):,}[/bold white] [dim]market caps retrieved[/dim]")
        console.print(f"    [dim]   {throughput:.0f} symbols/sec  •  {analysis_elapsed:.1f}s elapsed[/dim]")
        console.print()
        console.print()
        
        # ═══════════════════════════════════════════════════════════════════════════
        # PHASE 3: SELECTION
        # ═══════════════════════════════════════════════════════════════════════════
        phase3_title = Text()
        phase3_title.append("  3  ", style="bold black on white")
        phase3_title.append("  SELECTION", style="bold white")
        console.print(phase3_title)
        console.print()
        
        if len(ranked) < args.min_size:
            console.print(Panel(
                f"[bold red]Insufficient Valid Data[/bold red]\n\n"
                f"Only [bold]{len(ranked):,}[/bold] symbols have valid market caps.\n"
                f"Need [bold]{args.min_size:,}[/bold] for the target universe.",
                border_style="red",
                padding=(1, 4)
            ))
            sys.exit(3)
        
        selected = ranked[:args.min_size]
        tickers_out = [t for t, _ in selected]
        
        # Market cap distribution stats
        caps = [c for _, c in selected]
        min_cap = min(caps) if caps else 0
        max_cap = max(caps) if caps else 0
        median_cap = np.median(caps) if caps else 0
        p25_cap = np.percentile(caps, 25) if caps else 0
        p75_cap = np.percentile(caps, 75) if caps else 0
        
        console.print(f"    [green]●[/green]  [bold white]{len(selected):,}[/bold white] [dim]smallest by market cap selected[/dim]")
        console.print()
        
        # Beautiful stats display
        stats_panel_content = Table.grid(padding=(0, 3))
        stats_panel_content.add_column(justify="right", style="dim")
        stats_panel_content.add_column(justify="left", style="bold white")
        stats_panel_content.add_row("Smallest", human_money(min_cap))
        stats_panel_content.add_row("25th %ile", human_money(p25_cap))
        stats_panel_content.add_row("Median", human_money(median_cap))
        stats_panel_content.add_row("75th %ile", human_money(p75_cap))
        stats_panel_content.add_row("Largest", human_money(max_cap))
        
        console.print(Padding(
            Panel(
                Align.center(stats_panel_content),
                title="[dim]Market Cap Distribution[/dim]",
                border_style="dim",
                padding=(1, 4),
                box=box.ROUNDED
            ),
            (0, 4)
        ))
        console.print()
        console.print()
        
        # ═══════════════════════════════════════════════════════════════════════════
        # PHASE 4: SAVE
        # ═══════════════════════════════════════════════════════════════════════════
        phase4_title = Text()
        phase4_title.append("  4  ", style="bold black on white")
        phase4_title.append("  COMPLETE", style="bold white")
        console.print(phase4_title)
        console.print()
        
        out_path = args.out
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        
        df_out = pd.DataFrame({"ticker": tickers_out})
        df_out.to_csv(out_path, index=False)
        
        total_elapsed = time.time() - start_time_total
        
        console.print(f"    [green]●[/green]  Saved to [bold white]{out_path}[/bold white]")
        console.print()
        
        # ═══════════════════════════════════════════════════════════════════════════
        # PREVIEW TABLE - Top 10 smallest
        # ═══════════════════════════════════════════════════════════════════════════
        preview_table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="dim",
            padding=(0, 3),
            collapse_padding=True,
            show_edge=False,
        )
        preview_table.add_column("", justify="right", style="dim", width=3)
        preview_table.add_column("TICKER", style="bold white", width=10)
        preview_table.add_column("MARKET CAP", justify="right", style="dim", width=12)
        
        for i, (ticker, cap) in enumerate(selected[:8], 1):
            preview_table.add_row(str(i), ticker, human_money(cap))
        preview_table.add_row("⋮", "⋮", "⋮")
        preview_table.add_row(f"{len(selected):,}", selected[-1][0], human_money(selected[-1][1]))
        
        console.print(Padding(preview_table, (0, 4)))
        console.print()
        
        # ═══════════════════════════════════════════════════════════════════════════
        # FINAL SUMMARY - Clean footer
        # ═══════════════════════════════════════════════════════════════════════════
        console.print(Rule(style="dim"))
        console.print()
        
        summary_grid = Table.grid(padding=(0, 4))
        summary_grid.add_column(justify="center")
        summary_grid.add_column(justify="center")
        summary_grid.add_column(justify="center")
        summary_grid.add_row(
            f"[dim]Total Time[/dim]\n[bold white]{total_elapsed:.1f}s[/bold white]",
            f"[dim]Symbols[/dim]\n[bold bright_green]{len(tickers_out):,}[/bold bright_green]",
            f"[dim]Coverage[/dim]\n[bold white]{len(ranked)/len(candidates)*100:.0f}%[/bold white]",
        )
        console.print(Align.center(summary_grid))
        console.print()
        console.print(Align.center(Text("✓ Ready for screening", style="green")))
        console.print()
        
    else:
        # ═══════════════════════════════════════════════════════════════════════════
        # PLAIN TEXT FALLBACK - Still clean
        # ═══════════════════════════════════════════════════════════════════════════
        print()
        print("─" * 50)
        print("  R U S S E L L   5 0 0 0   B U I L D E R")
        print("─" * 50)
        print(f"  Cores: {num_processes}  •  Target: {args.min_size:,}")
        print()
        
        print("  [1] DISCOVERY")
        candidates = fetch_us_primary_listings_from_nasdaq_trader()
        
        if len(candidates) < MIN_EXPECTED_CANDIDATES:
            print(f"  ✗ Error: Only {len(candidates):,} candidates (need {MIN_EXPECTED_CANDIDATES:,}+)")
            sys.exit(2)
        print(f"      ● {len(candidates):,} symbols discovered")
        print()
        
        print("  [2] ANALYSIS")
        analysis_start = time.time()
        ranked = rank_by_market_cap_multiprocess(candidates, num_processes, use_rich=False)
        analysis_elapsed = time.time() - analysis_start
        print(f"      ● {len(ranked):,} market caps in {analysis_elapsed:.1f}s")
        print()
        
        print("  [3] SELECTION")
        if len(ranked) < args.min_size:
            print(f"  ✗ Error: Only {len(ranked):,} valid (need {args.min_size:,})")
            sys.exit(3)
        
        selected = ranked[:args.min_size]
        tickers_out = [t for t, _ in selected]
        print(f"      ● {len(selected):,} smallest selected")
        print()
        
        print("  [4] COMPLETE")
        out_path = args.out
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        
        df_out = pd.DataFrame({"ticker": tickers_out})
        df_out.to_csv(out_path, index=False)
        
        total_elapsed = time.time() - start_time_total
        print(f"      ● Saved to {out_path}")
        print()
        print("─" * 50)
        print(f"  {len(tickers_out):,} symbols  •  {total_elapsed:.1f}s total")
        print("─" * 50)
        print()


if __name__ == "__main__":
    main()
