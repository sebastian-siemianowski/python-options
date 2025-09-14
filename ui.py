"""
UI and formatting helpers for the options screener/backtester.

This module centralizes Rich console setup, progress iterators, and summary rendering.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

# Pretty console helpers (Rich) with graceful fallback
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.box import ROUNDED
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    _HAS_RICH = True
    # Force colors in common non-TTY contexts unless NO_COLOR is set.
    _force_color = str(os.environ.get("NO_COLOR", "")).strip() == ""
    _CON = Console(force_terminal=_force_color, color_system="auto")
except Exception:
    _HAS_RICH = False
    _CON = None


def _fmt_pct(x):
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "-"


def _progress_iter(seq, description=""):
    # Modern progress bar using Rich when available; falls back to sleek tqdm.
    try:
        total = len(seq)
    except Exception:
        total = None
    if _HAS_RICH:
        columns = [
            SpinnerColumn(style="cyan"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None, complete_style="bright_cyan", finished_style="green"),
            TextColumn("{task.completed}/{task.total}" if total is not None else "{task.completed}", style="white"),
            TextColumn("•", style="dim"),
            TimeElapsedColumn(),
            TextColumn("<", style="dim"),
            TimeRemainingColumn(),
        ]
        with Progress(*columns, transient=True, console=_CON) as progress:
            task_id = progress.add_task(description or "Working", total=total)
            for item in seq:
                yield item
                progress.advance(task_id)
    else:
        from tqdm import tqdm
        it = tqdm(
            seq,
            desc=description or "Working",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar:25} {n_fmt}/{total_fmt} • {elapsed}<{remaining} • {rate_fmt}",
            colour="cyan",
            mininterval=0.2,
        )
        for item in it:
            yield item


def _render_summary(tickers, df_res: pd.DataFrame, df_bt: pd.DataFrame):
    # Header printed at start; avoid duplicating here
    if df_res is not None and not df_res.empty:
        top = df_res[['ticker','expiry','dte','strike','mid','openInterest','volume','impliedVol','prob_10x']].head(10)
        if _HAS_RICH:
            from rich.table import Table
            tbl = Table(title="Top Option Candidates (preview)", box=ROUNDED, border_style="cyan")
            for col in top.columns:
                tbl.add_column(str(col), style="cyan" if col in ("ticker","expiry") else "white", justify="right")
            for _, r in top.iterrows():
                tbl.add_row(*[str(r[c]) for c in top.columns])
            _CON.print(tbl)
            _CON.print("Saved to [bold]screener_results.csv[/]", style="dim")
        else:
            print('Top results saved to screener_results.csv')
            print(top.to_string(index=False))

    if df_bt is not None and not df_bt.empty:
        # Backtest table (key metrics)
        cols = [
            'ticker','strategy_total_trades','strategy_win_rate','strategy_avg_trade_ret_x',
            'strategy_total_trade_profit_pct','strategy_CAGR','strategy_Sharpe','strategy_max_drawdown'
        ]
        # Create readable column name mapping
        col_names = {
            'ticker': 'Ticker',
            'strategy_total_trades': 'Total Trades', 
            'strategy_win_rate': 'Win Rate',
            'strategy_avg_trade_ret_x': 'Avg Trade Return',
            'strategy_total_trade_profit_pct': 'Total Trade Profit %',
            'strategy_CAGR': 'CAGR',
            'strategy_Sharpe': 'Sharpe Ratio',
            'strategy_max_drawdown': 'Max Drawdown'
        }
        present = [c for c in cols if c in df_bt.columns]
        prev = df_bt[present].copy()
        prev = prev.drop_duplicates(subset=['ticker']) if 'ticker' in prev.columns else prev
        if _HAS_RICH:
            # Add breathing room before the table
            _CON.print("")
            try:
                from rich.table import Table as _RichTable
            except Exception:
                _RichTable = None
            tbl2 = (_RichTable or Table)(title="Strategy Backtest Summary (per ticker)", box=ROUNDED, border_style="magenta")
            for c in present:
                display_name = col_names.get(c, c)
                tbl2.add_column(display_name, justify="right", style=("yellow" if c=="ticker" else "white"))
            for _, r in prev.head(20).iterrows():
                row_vals = []
                for c in present:
                    v = r[c]
                    if c in ('strategy_win_rate','strategy_CAGR','strategy_max_drawdown'):
                        row_vals.append(_fmt_pct(v))
                    elif c == 'strategy_total_trade_profit_pct':
                        row_vals.append(f"{float(v):.2f}%")
                    elif c == 'strategy_total_trades':
                        try:
                            iv = int(round(float(v)))
                            row_vals.append(f"{iv}")
                        except Exception:
                            row_vals.append(str(v))
                    else:
                        try:
                            row_vals.append(f"{float(v):.4f}")
                        except Exception:
                            row_vals.append(str(v))
                tbl2.add_row(*row_vals)
            _CON.print(tbl2)
            # And a blank line after
            _CON.print("")
        else:
            print(prev.head(20).to_string(index=False))

        # Spacer before combined profitability lines
        if _HAS_RICH:
            _CON.print("")
        else:
            print("")
        # Combined profitability lines
        try:
            combined_line = None
            avg_line = None
            if 'strategy_total_trades' in df_bt.columns and 'strategy_avg_trade_ret_x' in df_bt.columns:
                total_trades = float(df_bt['strategy_total_trades'].fillna(0).sum())
                if total_trades > 0:
                    weighted_avg_ret_x = (
                        (df_bt['strategy_avg_trade_ret_x'].fillna(0) * df_bt['strategy_total_trades'].fillna(0)).sum() / total_trades
                    )
                    combined_profit_pct = (weighted_avg_ret_x - 1.0) * 100.0
                    combined_line = f"Combined total profitability of all strategy trades: {combined_profit_pct:.2f}% (equal stake per trade)"
            if 'strategy_total_trade_profit_pct' in df_bt.columns:
                avg_pct = df_bt['strategy_total_trade_profit_pct'].dropna().mean()
                if np.isfinite(avg_pct):
                    avg_line = f"Average per-ticker total trade profitability: {avg_pct:.2f}%"
            if _HAS_RICH:
                if combined_line is not None:
                    style = "bold green" if ' -' not in combined_line and (combined_line.find(': ')!=-1 and float(combined_line.split(': ')[1].split('%')[0])>=0) else "bold red"
                    _CON.print(combined_line, style=style)
                if avg_line is not None:
                    avg_val = float(avg_line.split(': ')[1].split('%')[0])
                    style2 = "green" if avg_val>=0 else "red"
                    _CON.print(avg_line, style=style2)
                    _CON.print("")  # spacer after average profitability line
            else:
                if combined_line: print("\n" + combined_line)
                if avg_line:
                    print(avg_line)
                    print("")  # spacer after average profitability line
        except Exception:
            pass
