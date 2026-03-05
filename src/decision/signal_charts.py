"""
Signal Charts — 12-month candlestick charts with SMA overlays.

Two chart categories:
  1. SIGNAL CHARTS — Strong buy/sell signals from high-conviction JSONs
     → src/data/plots/signals/{TICKER}_signal.png
  2. SMA CHARTS — Stocks trading below their 50-day SMA
     → src/data/plots/sma/{TICKER}_sma.png

Both directories are cleared at the start of each run to keep charts fresh.
Called automatically after save_high_conviction_signals() in the signal flow.
"""

import json
import os
import pathlib
import re
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ─── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ─── Directories ─────────────────────────────────────────────────────────────
HC_BUY_DIR = os.path.join(REPO_ROOT, "src", "data", "high_conviction", "buy")
HC_SELL_DIR = os.path.join(REPO_ROOT, "src", "data", "high_conviction", "sell")
PRICE_DIR = os.path.join(REPO_ROOT, "src", "data", "prices")
CHARTS_DIR = os.path.join(REPO_ROOT, "src", "data", "plots", "signals")
SMA_CHARTS_DIR = os.path.join(REPO_ROOT, "src", "data", "plots", "sma")

# ─── Chart config ─────────────────────────────────────────────────────────────
CHART_LOOKBACK_MONTHS = 12
SMA_PERIODS = (10, 20, 50, 200)
SMA_COLORS = {
    10: "#FFD700",   # Gold/Yellow
    20: "#FF8C00",   # Dark Orange
    50: "#00CED1",   # Cyan
    200: "#DA70D6",  # Orchid/Magenta
}
FIGURE_SIZE = (14, 8)
DPI = 150

# ─── SMA50 filter config ─────────────────────────────────────────────────────
SMA50_PERIOD = 50
# Skip non-equity symbols (FX, indices, futures, crypto)
SMA50_SKIP_PATTERNS = re.compile(
    r"(.*_X$|^\^|.*_F$|.*-USD$|.*\.KS$|.*\.T$|.*\.L$|.*\.PA$|.*\.DE$)",
    re.IGNORECASE,
)

# ─── Lazy imports for mplfinance (avoid import cost when not charting) ────────
_mpf = None
_plt = None


def _ensure_mpf():
    """Lazy-load mplfinance and matplotlib."""
    global _mpf, _plt
    if _mpf is None:
        try:
            import mplfinance as mpf
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend for file saving
            import matplotlib.pyplot as plt
            _mpf = mpf
            _plt = plt
        except ImportError:
            raise ImportError(
                "mplfinance is required for signal charts. "
                "Install with: pip install mplfinance"
            )


def _price_csv_path(ticker: str) -> str:
    """
    Map ticker → price CSV path, matching data_utils._price_cache_path() sanitization.
    Replaces / = : with underscore, uppercases.
    """
    safe = ticker.replace("/", "_").replace("=", "_").replace(":", "_")
    return os.path.join(PRICE_DIR, f"{safe.upper()}.csv")


def _load_price_data(ticker: str, months: int = CHART_LOOKBACK_MONTHS) -> Optional[pd.DataFrame]:
    """
    Load OHLCV price data from CSV for the last N months.

    Returns a DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume
    suitable for mplfinance, or None if file not found / parsing fails.
    """
    csv_path = _price_csv_path(ticker)
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.set_index("Date")
        df = df.sort_index()

        # Filter to last N months
        cutoff = datetime.now() - timedelta(days=months * 30)
        df = df[df.index >= cutoff]

        if len(df) < 10:
            return None

        # Ensure correct column names for mplfinance
        rename_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower == "open":
                rename_map[col] = "Open"
            elif lower == "high":
                rename_map[col] = "High"
            elif lower == "low":
                rename_map[col] = "Low"
            elif lower in ("close", "adj close"):
                if "Close" not in rename_map.values():
                    rename_map[col] = "Close"
            elif lower == "volume":
                rename_map[col] = "Volume"

        df = df.rename(columns=rename_map)

        # Keep only what mplfinance needs
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None

        df = df[required].copy()

        # Drop rows with NaN in OHLC (Volume NaN is OK, fill with 0)
        df["Volume"] = df["Volume"].fillna(0)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])

        if len(df) < 10:
            return None

        return df

    except Exception:
        return None


def _get_signal_tickers() -> List[Dict]:
    """
    Read high-conviction manifests and collect unique tickers with signal info.

    Returns list of dicts:
      {"ticker": str, "asset_label": str, "signal_type": "STRONG_BUY"|"STRONG_SELL",
       "best_exp_ret": float, "best_probability": float}

    If a ticker appears in both buy and sell (different horizons), the one with
    the higher absolute expected return wins.
    """
    tickers: Dict[str, Dict] = {}

    for directory, default_type in [(HC_BUY_DIR, "STRONG_BUY"), (HC_SELL_DIR, "STRONG_SELL")]:
        if not os.path.isdir(directory):
            continue

        # Read individual signal JSONs (more reliable than manifest)
        for fname in os.listdir(directory):
            if not fname.endswith(".json") or fname == "manifest.json":
                continue

            fpath = os.path.join(directory, fname)
            try:
                with open(fpath, "r") as f:
                    sig = json.load(f)

                ticker = sig.get("ticker", "")
                if not ticker:
                    continue

                asset_label = sig.get("asset_label", ticker)
                signal_type = sig.get("signal_type", default_type)
                exp_ret = abs(sig.get("expected_return_pct", 0.0))
                prob = sig.get("probability_up", 0.5) if "BUY" in signal_type else sig.get("probability_down", 0.5)

                # Keep the signal with the highest absolute expected return per ticker
                if ticker in tickers:
                    if exp_ret > tickers[ticker]["best_exp_ret"]:
                        tickers[ticker] = {
                            "ticker": ticker,
                            "asset_label": asset_label,
                            "signal_type": signal_type,
                            "best_exp_ret": exp_ret,
                            "best_probability": prob,
                        }
                else:
                    tickers[ticker] = {
                        "ticker": ticker,
                        "asset_label": asset_label,
                        "signal_type": signal_type,
                        "best_exp_ret": exp_ret,
                        "best_probability": prob,
                    }

            except (json.JSONDecodeError, OSError):
                continue

    return list(tickers.values())


def _render_chart(
    ticker: str,
    asset_label: str,
    signal_type: str,
    price_df: pd.DataFrame,
    output_path: str,
) -> bool:
    """
    Render a candlestick chart with SMA overlays and save as PNG.

    Returns True on success, False on failure.
    """
    _ensure_mpf()

    try:
        is_buy = "BUY" in signal_type
        is_sma_warning = "BELOW_SMA" in signal_type
        is_reversal = signal_type == "REVERSAL_UP"

        if is_reversal:
            signal_color = "#00E676"   # Green — uptrend reversal
            signal_label = "↑ REVERSAL"
            signal_arrow = "↑"
        elif is_sma_warning:
            signal_color = "#FF6D00" if signal_type == "BELOW_SMA50" else "#FF1744"
            signal_label = "BELOW SMA50 + SMA200" if "200" in signal_type else "BELOW SMA50"
            signal_arrow = "▼▼" if "200" in signal_type else "▼"
        elif is_buy:
            signal_color = "#00E676"
            signal_label = "STRONG BUY"
            signal_arrow = "▲▲"
        else:
            signal_color = "#FF1744"
            signal_label = "STRONG SELL"
            signal_arrow = "▼▼"

        # ── Custom style ─────────────────────────────────────────────
        mc = _mpf.make_marketcolors(
            up="#26A69A",        # Teal green for up candles
            down="#EF5350",      # Red for down candles
            edge="inherit",
            wick="inherit",
            volume={"up": "#26A69A", "down": "#EF5350"},
            ohlc="inherit",
        )
        style = _mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mc,
            facecolor="#1a1a2e",
            edgecolor="#16213e",
            figcolor="#0f0f23",
            gridcolor="#1e2d4a",
            gridstyle="--",
            y_on_right=True,
            rc={
                "axes.labelcolor": "#b0b0b0",
                "xtick.color": "#808080",
                "ytick.color": "#808080",
            },
        )

        # ── SMA overlays ─────────────────────────────────────────────
        # Only include SMAs where we have enough data
        mav_periods = tuple(p for p in SMA_PERIODS if len(price_df) >= p)
        mav_colors = [SMA_COLORS[p] for p in mav_periods]

        # ── Title ─────────────────────────────────────────────────────
        title = f"{asset_label}  {signal_arrow} {signal_label}"

        # ── Plot ──────────────────────────────────────────────────────
        sma_legend_labels = [f"SMA {p}" for p in mav_periods]

        fig, axes = _mpf.plot(
            price_df,
            type="candle",
            style=style,
            title=title,
            volume=True,
            mav=mav_periods if mav_periods else None,
            mavcolors=mav_colors if mav_colors else None,
            figsize=FIGURE_SIZE,
            returnfig=True,
            panel_ratios=(4, 1),
            tight_layout=True,
            warn_too_much_data=9999,
        )

        # ── Style the title ──────────────────────────────────────────
        ax_main = axes[0]
        ax_main.set_title(title, fontsize=14, fontweight="bold", color=signal_color, pad=15)

        # ── SMA Legend ────────────────────────────────────────────────
        if mav_periods:
            from matplotlib.lines import Line2D
            legend_handles = [
                Line2D([0], [0], color=SMA_COLORS[p], linewidth=1.5, label=f"SMA {p}")
                for p in mav_periods
            ]
            ax_main.legend(
                handles=legend_handles,
                loc="upper left",
                fontsize=8,
                framealpha=0.7,
                facecolor="#1a1a2e",
                edgecolor="#333",
                labelcolor="#b0b0b0",
            )

        # ── Signal badge ──────────────────────────────────────────────
        ax_main.text(
            0.99, 0.97, signal_label,
            transform=ax_main.transAxes,
            fontsize=11, fontweight="bold",
            color=signal_color,
            ha="right", va="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#0f0f23",
                edgecolor=signal_color,
                alpha=0.9,
            ),
        )

        # ── Save ──────────────────────────────────────────────────────
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="#0f0f23")
        _plt.close(fig)
        return True

    except Exception:
        return False


def _clear_chart_dirs():
    """Clear both signal and SMA chart directories for fresh output."""
    for chart_dir in [CHARTS_DIR, SMA_CHARTS_DIR]:
        os.makedirs(chart_dir, exist_ok=True)
        for old_file in pathlib.Path(chart_dir).glob("*.png"):
            try:
                old_file.unlink()
            except OSError:
                pass


def generate_signal_charts(quiet: bool = False) -> Dict:
    """
    Generate candlestick charts for all strong buy/sell signals.

    Reads from src/data/high_conviction/{buy,sell}/ JSON files,
    loads 12 months of price data from src/data/prices/,
    and saves charts to src/data/plots/signals/.

    Both plots/signals/ and plots/sma/ are cleared at the start.

    Returns: {"generated": int, "skipped": int, "errors": int}
    """
    result = {"generated": 0, "skipped": 0, "errors": 0}

    # Always clear both chart directories for fresh output
    _clear_chart_dirs()

    # Collect signal tickers
    signals = _get_signal_tickers()
    if not signals:
        return result

    # Lazy-load mplfinance
    try:
        _ensure_mpf()
    except ImportError:
        if not quiet:
            try:
                from rich.console import Console
                Console().print("[yellow]⚠ mplfinance not installed — skipping signal charts[/yellow]")
            except Exception:
                pass
        result["skipped"] = len(signals)
        return result

    # Progress display
    console = None
    if not quiet:
        try:
            from rich.console import Console
            from rich.text import Text
            console = Console()

            header = Text()
            header.append("\n    ")
            header.append("📊 ", style="")
            header.append("Signal Charts", style="bold cyan")
            header.append(f"  ({len(signals)} charts)", style="dim")
            console.print(header)
        except Exception:
            pass

    # Generate one chart per ticker
    for i, sig in enumerate(signals, 1):
        ticker = sig["ticker"]
        asset_label = sig["asset_label"]
        signal_type = sig["signal_type"]

        # Load price data
        price_df = _load_price_data(ticker)
        if price_df is None:
            result["skipped"] += 1
            if console:
                console.print(f"    [dim]  skip {ticker} (no price data)[/dim]")
            continue

        # Output path
        safe_ticker = ticker.replace("/", "_").replace("=", "_").replace(":", "_").replace("^", "_")
        output_path = os.path.join(CHARTS_DIR, f"{safe_ticker}_signal.png")

        # Render
        success = _render_chart(ticker, asset_label, signal_type, price_df, output_path)

        if success:
            result["generated"] += 1
            if console:
                signal_icon = "▲" if "BUY" in signal_type else "▼"
                color = "green" if "BUY" in signal_type else "red"
                console.print(f"    [dim]  ✓[/dim] [{color}]{signal_icon}[/{color}] [dim]{ticker}[/dim]")
        else:
            result["errors"] += 1
            if console:
                console.print(f"    [dim]  ✗ {ticker} (chart error)[/dim]")

    # Summary
    if console and result["generated"] > 0:
        from rich.text import Text
        summary = Text()
        summary.append("    ")
        summary.append(f"→ {result['generated']} charts", style="bold green")
        summary.append(f" saved to src/data/plots/signals/", style="dim")
        console.print(summary)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# BELOW SMA50 DETECTION — Scan price universe for stocks under their 50-day SMA
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_ticker_from_label(asset_label: str) -> str:
    """Extract ticker from asset label like 'Apple Inc. (AAPL)' → 'AAPL'."""
    match = re.search(r'\(([A-Z0-9.=^-]+)\)', asset_label)
    return match.group(1) if match else asset_label.split()[0] if asset_label else ""


def find_below_sma50_stocks(
    summary_rows: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Find stocks trading below their 50-day SMA.

    If summary_rows is provided, only checks tickers from those rows.
    Otherwise, scans all equity CSVs in src/data/prices/.

    Returns list of dicts sorted by deviation (most below SMA50 first):
      {"ticker": str, "asset_label": str, "sector": str,
       "close": float, "sma50": float, "deviation_pct": float,
       "sma200": float|None, "below_sma200": bool}
    """
    results = []

    if summary_rows:
        # Use tickers from the signal run
        ticker_info = {}
        for row in summary_rows:
            asset_label = row.get("asset_label", "")
            sector = row.get("sector", "Other")
            ticker = _extract_ticker_from_label(asset_label)
            if ticker and ticker not in ticker_info:
                ticker_info[ticker] = {"asset_label": asset_label, "sector": sector}

        candidates = list(ticker_info.items())
    else:
        # Scan all price CSVs
        candidates = []
        if not os.path.isdir(PRICE_DIR):
            return results
        for fname in sorted(os.listdir(PRICE_DIR)):
            if not fname.endswith(".csv"):
                continue
            ticker = fname.replace(".csv", "")
            # Skip non-equity symbols
            if SMA50_SKIP_PATTERNS.match(ticker):
                continue
            candidates.append((ticker, {"asset_label": ticker, "sector": ""}))

    for ticker, info in candidates:
        # Skip non-equity symbols when from summary_rows too
        safe_ticker = ticker.replace("/", "_").replace("=", "_").replace(":", "_")
        if SMA50_SKIP_PATTERNS.match(safe_ticker):
            continue

        csv_path = _price_csv_path(ticker)
        if not os.path.exists(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            df = df.sort_values("Date")

            # Need column names
            close_col = None
            for col in df.columns:
                if col.lower().strip() in ("close", "adj close"):
                    close_col = col
                    break
            if close_col is None:
                continue

            closes = df[close_col].dropna()
            if len(closes) < SMA50_PERIOD:
                continue

            current_close = closes.iloc[-1]
            sma50 = closes.iloc[-SMA50_PERIOD:].mean()

            if current_close >= sma50:
                continue  # Not below SMA50

            deviation_pct = ((current_close - sma50) / sma50) * 100.0

            # Also compute SMA200 if available
            sma200 = None
            below_sma200 = False
            if len(closes) >= 200:
                sma200 = closes.iloc[-200:].mean()
                below_sma200 = current_close < sma200

            # ── Uptrend reversal detection ────────────────────────────
            # Stock is below SMA50 but showing signs of turning up:
            #   1. Close > SMA10 (short-term uptrend)
            #   2. 5-day return is positive
            #   3. SMA10 slope is positive (10-day MA rising)
            reversing_up = False
            sma10 = None
            ret_5d = None
            if len(closes) >= 10:
                sma10 = closes.iloc[-10:].mean()
                sma10_prev = closes.iloc[-11:-1].mean() if len(closes) >= 11 else sma10
                sma10_rising = sma10 > sma10_prev
                above_sma10 = current_close > sma10
                ret_5d = ((current_close / closes.iloc[-6]) - 1.0) * 100.0 if len(closes) >= 6 else 0.0
                reversing_up = above_sma10 and sma10_rising and ret_5d > 0

            results.append({
                "ticker": ticker,
                "asset_label": info["asset_label"],
                "sector": info.get("sector", ""),
                "close": round(current_close, 2),
                "sma50": round(sma50, 2),
                "sma10": round(sma10, 2) if sma10 is not None else None,
                "deviation_pct": round(deviation_pct, 2),
                "sma200": round(sma200, 2) if sma200 is not None else None,
                "below_sma200": below_sma200,
                "reversing_up": reversing_up,
                "ret_5d": round(ret_5d, 2) if ret_5d is not None else None,
            })

        except Exception:
            continue

    # Sort by deviation (most below SMA50 first)
    results.sort(key=lambda x: x["deviation_pct"])
    return results


def render_below_sma50_table(
    below_sma50: List[Dict],
    quiet: bool = False,
) -> None:
    """
    Render a Rich table of stocks trading below their 50-day SMA.
    """
    if not below_sma50 or quiet:
        return

    # Filter to only reversing-up stocks for display
    reversing_stocks = [s for s in below_sma50 if s.get("reversing_up")]
    if not reversing_stocks:
        return

    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
        from rich.panel import Panel
        from rich import box
    except ImportError:
        return

    console = Console()

    # Header
    console.print()
    header = Text()
    header.append("  ↑ SMA50 REVERSALS", style="bold green")
    header.append(f"  ({len(reversing_stocks)} of {len(below_sma50)} below SMA50)", style="dim")
    console.print(Panel(header, box=box.HEAVY, border_style="green", expand=False))

    # Table
    table = Table(
        box=box.ROUNDED,
        border_style="green",
        header_style="bold white on dark_green",
        row_styles=["", "on grey7"],
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("Asset", width=50, no_wrap=True)
    table.add_column("Sector", width=28, no_wrap=True)
    table.add_column("Close", justify="right", width=10)
    table.add_column("SMA50", justify="right", width=10)
    table.add_column("Dev %", justify="right", width=10)
    table.add_column("SMA200", justify="right", width=10)
    table.add_column("Status", width=14)
    table.add_column("5d Return", width=12)

    for stock in reversing_stocks[:40]:  # Cap at 40
        # Deviation color: deeper red for more deviation
        dev = stock["deviation_pct"]
        if dev <= -20:
            dev_style = "bold red"
        elif dev <= -10:
            dev_style = "red"
        elif dev <= -5:
            dev_style = "bright_red"
        else:
            dev_style = "yellow"

        # Status badge
        if stock["below_sma200"]:
            status = Text("▼▼ WEAK", style="bold red")
        else:
            status = Text("▼ CAUTION", style="yellow")

        # 5-day return
        ret_5d = stock.get("ret_5d", 0)
        ret_text = Text(f"↑ +{ret_5d:.1f}%", style="bold green")

        sma200_str = f"${stock['sma200']:,.2f}" if stock["sma200"] is not None else "—"

        table.add_row(
            stock["asset_label"],
            stock["sector"],
            f"${stock['close']:,.2f}",
            f"${stock['sma50']:,.2f}",
            Text(f"{dev:+.1f}%", style=dev_style),
            sma200_str,
            status,
            ret_text,
        )

    console.print(table)

    # Footer
    below_both = sum(1 for s in reversing_stocks if s["below_sma200"])
    footer = Text()
    footer.append("    ")
    footer.append(f"Reversing: {len(reversing_stocks)}", style="bold green")
    footer.append("  ·  ", style="dim")
    footer.append(f"Also below SMA200: {below_both}", style="red")
    footer.append("  ·  ", style="dim")
    footer.append(f"Total below SMA50: {len(below_sma50)}", style="dim")
    console.print(footer)


def generate_sma_charts(
    below_sma50: List[Dict],
    quiet: bool = False,
) -> Dict:
    """
    Generate candlestick charts for stocks below SMA50 that show uptrend reversal.

    Only charts stocks where:
      - Close > SMA10 (short-term recovery)
      - SMA10 is rising (momentum turning)
      - 5-day return is positive

    Saves to src/data/plots/sma/{TICKER}_sma.png

    Returns: {"generated": int, "skipped": int, "errors": int, "reversing": int}
    """
    result = {"generated": 0, "skipped": 0, "errors": 0, "reversing": 0}

    if not below_sma50:
        return result

    # Filter to only reversing-up stocks for charting
    reversing_stocks = [s for s in below_sma50 if s.get("reversing_up")]
    result["reversing"] = len(reversing_stocks)

    if not reversing_stocks:
        return result

    # Clear and recreate SMA chart directory for fresh output
    os.makedirs(SMA_CHARTS_DIR, exist_ok=True)
    for old_file in pathlib.Path(SMA_CHARTS_DIR).glob("*.png"):
        try:
            old_file.unlink()
        except OSError:
            pass

    # Lazy-load mplfinance
    try:
        _ensure_mpf()
    except ImportError:
        if not quiet:
            try:
                from rich.console import Console
                Console().print("[yellow]⚠ mplfinance not installed — skipping SMA charts[/yellow]")
            except Exception:
                pass
        result["skipped"] = len(reversing_stocks)
        return result

    # Progress display
    console = None
    if not quiet:
        try:
            from rich.console import Console
            from rich.text import Text
            console = Console()

            header = Text()
            header.append("\n    ")
            header.append("📈 ", style="")
            header.append("SMA Reversal Charts", style="bold green")
            header.append(f"  ({len(reversing_stocks)} reversing up)", style="dim")
            console.print(header)
        except Exception:
            pass

    for stock in reversing_stocks:
        ticker = stock["ticker"]
        asset_label = stock["asset_label"]

        price_df = _load_price_data(ticker)
        if price_df is None:
            result["skipped"] += 1
            continue

        signal_type = "REVERSAL_UP"

        safe_ticker = ticker.replace("/", "_").replace("=", "_").replace(":", "_").replace("^", "_")
        output_path = os.path.join(SMA_CHARTS_DIR, f"{safe_ticker}_sma.png")

        success = _render_chart(ticker, asset_label, signal_type, price_df, output_path)

        if success:
            result["generated"] += 1
            if console:
                dev = stock["deviation_pct"]
                ret_5d = stock.get("ret_5d", 0)
                console.print(f"    [dim]  ✓[/dim] [green]↑[/green] [dim]{ticker} (SMA50: {dev:+.1f}%, 5d: +{ret_5d:.1f}%)[/dim]")
        else:
            result["errors"] += 1

    # Summary
    if console and result["generated"] > 0:
        from rich.text import Text
        summary = Text()
        summary.append("    ")
        summary.append(f"→ {result['generated']} charts", style="bold green")
        summary.append(f" saved to src/data/plots/sma/", style="dim")
        console.print(summary)

    return result


# ─── CLI entry point for standalone usage ─────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate signal charts from high-conviction signals")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    parser.add_argument("--sma-only", action="store_true", help="Only generate SMA50 charts")
    parser.add_argument("--signals-only", action="store_true", help="Only generate signal charts")
    args = parser.parse_args()

    if not args.sma_only:
        result = generate_signal_charts(quiet=args.quiet)
        if not args.quiet:
            print(f"\nSignal Charts: {result['generated']} generated, "
                  f"{result['skipped']} skipped, {result['errors']} errors")

    if not args.signals_only:
        below = find_below_sma50_stocks()
        if not args.quiet:
            render_below_sma50_table(below)
        sma_result = generate_sma_charts(below, quiet=args.quiet)
        if not args.quiet:
            print(f"\nSMA Charts: {sma_result['generated']} generated, "
                  f"{sma_result['skipped']} skipped, {sma_result['errors']} errors")
