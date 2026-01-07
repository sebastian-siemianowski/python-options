#!/usr/bin/env python3
"""
backtest_signals.py

A world-class, architected backtesting framework for evaluating the performance
of the signal generation logic from fx_pln_jpy_signals.py.
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.fx_data_utils import get_default_asset_universe
from scripts.fx_pln_jpy_signals import compute_features, generate_historical_signals

def run_backtest(price_series: pd.Series, signal_series: pd.DataFrame, t_costs: float) -> pd.DataFrame:
    """
    Core backtesting engine.

    Args:
        price_series: Series of historical prices.
        signal_series: DataFrame with 'date', 'signal_label', and 'pos_strength'.
        t_costs: Transaction costs as a fraction.

    Returns:
        DataFrame representing the equity curve.
    """
    signals = signal_series.set_index('date')

    # Align price and signal data
    df = pd.DataFrame({'price': price_series})
    df = df.join(signals, how='left')
    df['pos_strength'] = df['pos_strength'].ffill().fillna(0)

    df['returns'] = df['price'].pct_change()

    # Calculate target position based on signal
    df['target_position'] = np.where(df['signal_label'] == 'BUY', df['pos_strength'],
                                     np.where(df['signal_label'] == 'SELL', -df['pos_strength'], 0))

    # Lag positions to avoid look-ahead bias
    df['position'] = df['target_position'].shift(1).fillna(0)

    # Calculate transaction costs
    df['trades'] = df['position'].diff().abs()
    df['t_costs'] = df['trades'] * t_costs

    # Calculate strategy returns
    df['strategy_returns'] = df['position'] * df['returns'] - df['t_costs']
    df['strategy_returns'] = df['strategy_returns'].fillna(0)

    # Calculate equity curve
    df['equity_curve'] = (1 + df['strategy_returns']).cumprod()

    return df

def calculate_performance_metrics(equity_curve: pd.DataFrame) -> Dict:
    """
    Calculate performance metrics from an equity curve.

    Args:
        equity_curve: DataFrame with an 'equity_curve' column.

    Returns:
        Dictionary of performance metrics.
    """
    total_return = equity_curve['equity_curve'].iloc[-1] - 1
    n_days = len(equity_curve)
    cagr = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

    annualized_vol = equity_curve['strategy_returns'].std() * np.sqrt(252)
    sharpe_ratio = cagr / annualized_vol if annualized_vol > 0 else 0

    # Max drawdown
    rolling_max = equity_curve['equity_curve'].cummax()
    drawdown = (equity_curve['equity_curve'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Win rate and profit factor
    trade_returns = equity_curve[equity_curve['trades'] > 0]['strategy_returns']
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else np.inf

    return {
        'cagr': cagr,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_return': total_return,
        'n_trades': len(trade_returns),
    }

def save_report(performance_metrics: Dict, asset: str, reports_dir: str):
    """
    Saves the performance metrics report to a JSON file.

    Args:
        performance_metrics: Dictionary of performance metrics.
        asset: The asset symbol.
        reports_dir: Directory to save the report.
    """
    # Convert numpy types to native Python types for JSON serialization
    for key, value in performance_metrics.items():
        if isinstance(value, np.integer):
            performance_metrics[key] = int(value)
        elif isinstance(value, np.floating):
            performance_metrics[key] = float(value)
        elif isinstance(value, np.ndarray):
            performance_metrics[key] = value.tolist()

    report_path = os.path.join(reports_dir, f"{asset}_performance_report.json")
    with open(report_path, 'w') as f:
        json.dump(performance_metrics, f, indent=4)
    print(f"Performance report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run a world-class backtest on the signal generation logic.")
    parser.add_argument('--assets', type=str, default=",".join(get_default_asset_universe()), help='Comma-separated list of asset symbols to backtest.')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start date for historical data.')
    parser.add_argument('--end', type=str, default=None, help='End date for historical data.')
    parser.add_argument('--t-costs', type=float, default=0.0005, help='Transaction costs (e.g., 0.0005 for 5 bps).')
    parser.add_argument('--reports-dir', type=str, default='backtests/reports', help='Directory to save JSON reports.')
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(',') if a.strip()]
    os.makedirs(args.reports_dir, exist_ok=True)

    print("Starting world-class backtesting...")

    for asset in assets:
        print(f"\nBacktesting for asset: {asset}")
        try:
            # Generate signals
            signals_df = generate_historical_signals(asset, args.start, args.end)

            # Get price data
            from fx_pln_jpy_signals import fetch_px_asset
            price_series, _ = fetch_px_asset(asset, args.start, args.end)

            # Run the backtest
            equity_curve = run_backtest(price_series, signals_df, args.t_costs)

            # Calculate performance metrics
            performance_metrics = calculate_performance_metrics(equity_curve)

            # Save the report
            save_report(performance_metrics, asset, args.reports_dir)

        except Exception as e:
            print(f"  ! An error occurred while backtesting {asset}: {e}")
            continue

    print("\nWorld-class backtesting complete.")

if __name__ == '__main__':
    main()
